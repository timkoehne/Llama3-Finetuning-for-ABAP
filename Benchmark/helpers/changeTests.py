import json
import os

with open("human-eval-v2-20210705.json") as file:
    human_eval = json.loads(file.read())


for index, entry in enumerate(human_eval):
    newContent = []

    content = entry["test"]
    num_asserts = content.count("assert")
    replace_with = f"""
def num_tests():
    return {num_asserts}

def check(candidate):"""
    content = content.replace("def check(candidate):", replace_with)

    content = content.splitlines()

    for line in content:
        indent = len(line) - len(line.lstrip())
        if "def check" in line:
            newContent.append(line)
            newContent.append(" " * (indent + 4) + "passed = 0")
            newContent.append(" " * (indent + 4) + "failed = 0")
        elif "assert True" in line:
            # remove assert True
            pass
        elif "assert" in line:
            newContent.append(" " * indent + "try:")
            newContent.append(" " * 4 + line)
            newContent.append(" " * (indent + 4) + "passed += 1")
            newContent.append(" " * indent + "except (AssertionError, TypeError):")
            newContent.append(" " * (indent + 4) + "failed += 1")
        else:
            newContent.append(line)
    newContent.append("    return passed, failed")

    with open("adjustedTests/test" + str(index) + ".py", "w") as file:
        file.writelines([line + "\n" for line in newContent])
