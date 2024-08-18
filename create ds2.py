import json
import pickle
from queue import Queue
from random import shuffle
from threading import Thread
from time import sleep
from typing import Dict, List
from datasets import load_dataset

import tqdm
from ollamaWrapper import Ollama

ollama = Ollama("You translate code into abap", "llama3.1:8b-instruct-q4_K_M")
AMOUNT_OF_EXAMPLES = 1000

translated = []

def _extract_code(text: str) -> str:
    if "```" in text:
        start_index = text.index("```")
        if "```" in text[start_index+1:]:
            code_start = text.index("\n", start_index) + 1
            code_end = text.index("```", code_start)
            text = text[code_start:code_end]
    return text


def _threaded_ask_for_translation(q: Queue, thread_id):
    print(f"thread {thread_id} started")
    while q.unfinished_tasks > 0:
        entry = q.get()
        if entry is None:
            q.task_done()
            return

        translation = ollama.askWithoutContext(entry["output"])
        entry["abap"] = _extract_code(translation)
        translated.append(entry)
        # print(f"thread {thread_id} finished a task. remaining {q.unfinished_tasks}")
        q.task_done()


def ask_for_translation():
    queue = Queue()
    num_threads = 5

    ds = load_dataset("HuggingFaceH4/CodeAlpaca_20K", split="train[:1000]")
    ds_size = len(list(ds))

    for entry in ds:
        queue.put(entry)

    workers: list[Thread] = []
    for i in range(0, num_threads):
        worker = Thread(target=_threaded_ask_for_translation, args=(queue, i))
        workers.append(worker)
        worker.start()

    progress_bar = tqdm.tqdm(range(ds_size))
    while queue.unfinished_tasks > 0:
        progress_bar.n = ds_size - queue.unfinished_tasks
        progress_bar.refresh()
        sleep(1)
    queue.join()
    progress_bar.n = ds_size
    progress_bar.refresh()
    print("done")

    with open("translated.json", "w") as file:  # function_module
        file.write(json.dumps(translated, indent=4))


def formatting_for_fine_tuning():
    content = []
    with open("translated.json", "r") as file:
        data = json.loads(file.read())
        content = [{"instruction": entry["instruction"], "input": entry["input"], "output": entry["abap"]} for entry in data ]

    with open("DS2.json", "w") as file:
        file.write(json.dumps(content, indent=4))


# ask_for_translation()
# formatting_for_fine_tuning()