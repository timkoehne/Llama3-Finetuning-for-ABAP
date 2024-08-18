
import glob
import os
import time
import rfcMethod

# #-----------big run------------
systemPromptOptions = ["generate an abap function module. no explanations or examples."]
promptOptions = ["prompts/"]
temperatureOptions = [1] #ignored
promptsToRun = [i for i in range(0, 164)] #i for i in range(0, 164)
numRepeats = 100
numThreads = 5

filename = "ds1-v1"
functiongroup = "ZRFCTEST23"


# # generate answers
# rfcMethod.askLLMForPromptsMultithreaded(f"results/{filename}.json", systemPromptOptions, promptOptions, temperatureOptions, f"{filename}-llama3.1:8b-Q4_K_M", 0.5, promptsToRun, numRepeats, numThreads)



# # start running functions
# rfcMethod.runSavedFunctions(f"results/{filename}.json", f"results/{filename}-Processed.json", functiongroup, 0)


# # continue running functions after crash
# list_of_files = glob.glob(f'backup\\results\\{filename}*')
# latest_file = max(list_of_files, key=os.path.getmtime)
# left_off_at_pos = rfcMethod.find_left_off_position(latest_file)
# print(f"Last run stopped running at {left_off_at_pos}")
# rfcMethod.runSavedFunctions(latest_file, f"results/{filename}-Processed.json", functiongroup, left_off_at_pos) #continue after crash
