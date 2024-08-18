import json
import pickle
from queue import Queue
import re
import os
from threading import Thread
from time import sleep
from typing import Dict, List
import boto3
from smart_open import open
from datasets import (
    load_dataset,
    load_from_disk,
    DatasetDict,
    Dataset,
    IterableDatasetDict,
    IterableDataset,
)
import tqdm
import ollamaWrapper
from transformers import AutoTokenizer

# downloading ds1 requires you to login to both aws-cli and huggingface-cli
# You also need to accept the dataset conditions on huggingface for bigcode/the-stack-v2


NUM_THREADS_LLM_REQUEST = 10
LLM_MODEL = "llama3.1:8b-instruct-q4_K_M"
FILTER_OUT_PHRASES_INCLUDES = [
    "```" "given code" "provided code",
    "This appears",
    "it appears",
    "the provided code",
    "The code you provided",
    "This code appears",
    "It looks like you",
]

session = boto3.Session()
s3 = session.client("s3")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")

ollama = ollamaWrapper.Ollama("", LLM_MODEL)

def download_contents(blob_id, src_encoding):
    s3_url = f"s3://softwareheritage/content/{blob_id}"

    with open(s3_url, "rb", compression=".gz", transport_params={"client": s3}) as fin:
        content = fin.read().decode(src_encoding)

    return {"content": content}


def download_dataset() -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    print("downloading dataset...")
    ds = load_dataset("bigcode/the-stack-v2", "ABAP", split="train")
    ds = ds.map(lambda row: download_contents(row["blob_id"], row["src_encoding"]))
    return ds


def classify_abap_file(file_content: str) -> str:

    file_content_lines = file_content.splitlines()
    file_content_lines = [
        line
        for line in file_content_lines
        if not line.strip().startswith("*") and not line.strip().startswith('"')
    ]
    file_content = "\n".join(file_content_lines)

    # Define regex patterns for different ABAP file types
    abap_variable_name_regex = r"(/[a-zA-Z0-9_]+/)?[a-zA-Z][a-zA-Z0-9_]{0,28}"
    patterns = {
        "Function Module": re.compile(
            r"^\s*FUNCTION\s+" + abap_variable_name_regex, re.IGNORECASE
        ),
        "Class": re.compile(r"^\s*CLASS\s+" + abap_variable_name_regex, re.IGNORECASE),
        "Report": re.compile(
            r"^\s*REPORT\s+" + abap_variable_name_regex, re.IGNORECASE
        ),
        "Include": re.compile(
            r"^\s*INCLUDE\s+" + abap_variable_name_regex, re.IGNORECASE
        ),
        "Interface": re.compile(
            r"^\s*INTERFACE\s" + abap_variable_name_regex, re.IGNORECASE
        ),
        "Method": re.compile(
            r"^\s*METHOD\s+" + abap_variable_name_regex, re.IGNORECASE
        ),
        "Form": re.compile(r"^\s*FORM\s+" + abap_variable_name_regex, re.IGNORECASE),
        "Module": re.compile(r"^\s*MODULE\s+\w+", re.IGNORECASE),
    }

    # Check each pattern
    for file_type, pattern in patterns.items():
        if pattern.search(file_content):
            return file_type

    # If no pattern matches
    return "Unknown"


def sort_dataset_into_categories(ds):
    sorting_dict = {}
    print("sorting abap into file categories...")
    for i in tqdm.tqdm(ds):
        code = str(i["content"])  # type: ignore
        category = classify_abap_file(code)
        if category not in sorting_dict:
            sorting_dict[category] = []
        sorting_dict[category].append(code)
    return sorting_dict


exercise_data = []
def _threaded_ask_for_multiple_exercises(q: Queue, thread_id):
    while q.unfinished_tasks > 0:
        solution = q.get()
        if solution is None:
            q.task_done()
            return

        messages = [
            """I will give you some abap code, use this to write a brief excercise so that my code is the solution to your exercise.
You do not need to explain every variable. 
Just write the exercise, do not address this message""",
            solution,
        ]
        exercise = ollama.askWithoutContext(messages)
        exercise_data.append({"exercise": exercise, "solution": solution})
        # print(f"thread {thread_id} finished a task. remaining {q.unfinished_tasks}")
        q.task_done()


def generate_excercises(abap_solutions):
    queue = Queue()
    print("generating excercises...")

    for abap_code in abap_solutions:
        queue.put(abap_code)

    workers: list[Thread] = []
    for i in range(0, NUM_THREADS_LLM_REQUEST):
        worker = Thread(target=_threaded_ask_for_multiple_exercises, args=(queue, i))
        workers.append(worker)
        worker.start()

    progress_bar = tqdm.tqdm(range(len(abap_solutions)))
    while queue.unfinished_tasks > 0:
        progress_bar.n = len(abap_solutions) - queue.unfinished_tasks
        progress_bar.refresh()
        sleep(1)
    queue.join()
    progress_bar.n = queue.unfinished_tasks
    progress_bar.refresh()

    return exercise_data


def _count_tokens(text: str):
    tokens = tokenizer.tokenize(text)
    num_tokens = len(tokens)
    return num_tokens


def _filter_out_phrases(entry: dict[str, str]):
    for phrase in FILTER_OUT_PHRASES_INCLUDES:
        if phrase.lower() in entry["prompt"].lower():
            return True
    return False


def filter_data(all_data):
    filtered_data = []
    num_skipped_content = 0
    num_skipped_length = 0
    num_edited = 0
    for entry in tqdm.tqdm(all_data):
        edited_entry = {}
        edited_entry["prompt"] = entry["exercise"]
        edited_entry["response"] = entry["solution"]

        if (
            _count_tokens(edited_entry["prompt"]) > 8192
            or _count_tokens(edited_entry["response"]) > 8192
        ):
            num_skipped_length += 1
            continue

        # remove entries
        if _filter_out_phrases(edited_entry):
            num_skipped_content += 1
            continue

        # edit entries
        if (
            edited_entry["prompt"].startswith("** Exercise:")
            or edited_entry["prompt"].startswith("**Exercise:")
            or edited_entry["prompt"].startswith("Exercise:")
        ):
            num_edited += 1
            edited_entry["prompt"] = edited_entry["prompt"][
                edited_entry["prompt"].index(":") + 1 :
            ]
        edited_entry["prompt"] = edited_entry["prompt"].strip()
        filtered_data.append(edited_entry)

    print(f"there are {len(all_data)} overall")
    print(f"filtering {num_skipped_content} because of content")
    print(f"filtering {num_skipped_length} because of length")
    print(f"modifying {num_skipped_length} entries")
    print(f"there are {len(filtered_data)} remaining")
    return filtered_data


if os.path.isdir("abap_code_ds"):
    ds = load_from_disk("abap_code_ds")
else:
    ds = download_dataset()
    ds.save_to_disk("abap_code_ds") # type: ignore
    
ds = list(ds)
sorted = sort_dataset_into_categories(ds)

abap_solutions = []
if "Function Module" in sorted: 
    abap_solutions += sorted["Function Module"]
if "Class" in sorted: 
    abap_solutions += sorted["Class"]
if "Report" in sorted: 
    abap_solutions += sorted["Report"]

ds1 = generate_excercises(abap_solutions)
ds1 = filter_data(ds1)

with open("DS1.json", "w") as file:
    file.write(json.dumps(ds1, indent=4))
