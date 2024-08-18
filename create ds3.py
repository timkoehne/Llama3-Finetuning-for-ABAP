import json
from datasets import load_dataset

ds = load_dataset("smjain/abap", split="train")

with open("DS3.json", "w") as file:
    file.write(json.dumps(list(ds), indent=4))
