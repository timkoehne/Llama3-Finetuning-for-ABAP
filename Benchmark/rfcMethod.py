import glob
import importlib
import json
import os
import sys
import more_itertools
import pyrfc
import threading
import queue
import time
import pythonAbapInterface
import re
import time

# from chatGptWrapper import ChatGPT
from ollamaWrapper import Ollama


class Test:
    def __init__(
        self,
        conn,
        functionCreated,
        functionName,
        importParameters,
        exportParameters,
        saveTo,
    ) -> None:
        self.conn = conn
        self.functionCreated = functionCreated
        self.functionName = functionName
        self.importParameters = importParameters
        self.exportParameters = exportParameters
        self.saveTo = saveTo
        self.saveTo["functionCalls"] = []

    def callFunction(self, *args):

        if self.functionCreated == "success":
            callParams = buildAbapImportParameters(args, self.importParameters)
            result = pythonAbapInterface.callFunctionModule(
                self.conn, self.functionName, callParams
            )
        else:
            result = {"exception": "could not run because function was not created"}

        self.saveTo["functionCalls"].append(result)
        if len(result) > 0:
            result = result[list(result)[0]]
        return result


class AbapVariable:
    def __init__(self, abapCodeLine: str) -> None:
        self.varName = (
            abapCodeLine.split(" ")[0]
            .replace("VALUE(", "")
            .replace(")", "")
            .upper()
            .removesuffix(".")
        )
        self.varType = (
            " ".join(" ".join(abapCodeLine.split()).split()[2:])
            .upper()
            .removesuffix(".")
        )


class LLMRequest:
    def __init__(self, systemPrompt, promptOption, temperature, promptNumber) -> None:
        self.systemPrompt = systemPrompt
        self.promptOption = promptOption
        self.temperature = temperature
        self.promptNumber = promptNumber


class LLMResponse:
    def __init__(self, request: LLMRequest, llmResponse: list[str]) -> None:
        self.request: LLMRequest = request
        self.llmResponse: list[str] = llmResponse

    def __str__(self) -> str:
        return (
            self.request.systemPrompt
            + " "
            + self.request.promptOption
            + " "
            + str(self.request.temperature)
            + " "
            + str(self.request.promptNumber)
            + " "
            + self.llmResponse
        )


def parseAbapFunctionInterface(llmResponse: list[str]):
    interface = [line for line in llmResponse if line.startswith("*")]
    interface = [line.replace('*"', "").strip() for line in interface]
    interface = [line for line in interface if line.strip("-")]
    imports = []
    exports = []
    state = None
    for line in interface:
        if line == "IMPORTING" or line == "EXPORTING":
            state = line
            continue
        if state == "IMPORTING":
            imports.append(line)
        elif state == "EXPORTING":
            exports.append(line)

    importVariables = [AbapVariable(line) for line in imports]
    exportVariables = [AbapVariable(line) for line in exports]

    importParameters = [
        {"PARAMETER": parameter.varName, "TYP": parameter.varType}
        for parameter in importVariables
    ]
    exportParameters = [
        {"PARAMETER": parameter.varName, "TYP": parameter.varType}
        for parameter in exportVariables
    ]

    return importParameters, exportParameters


def buildAbapImportParameters(params, importParameters):
    callParams = {}

    smaller = (
        len(params) if len(params) <= len(importParameters) else len(importParameters)
    )
    for index in range(0, smaller):
        key = list(importParameters)[index]
        callParams[key["PARAMETER"]] = params[index]

    return callParams


def getPrompt(folderPath: str, promptNumber: int):
    with open(folderPath + str(promptNumber) + ".txt") as file:
        prompt = file.read()
        return prompt


def extract_code(text: str) -> str:
    if "```" in text:
        start_index = text.index("```")
        code_start = text.index("\n", start_index) + 1
        code_end = text.index("```", code_start)
        text = text[code_start:code_end]
    return text


def extractAbapFunctionInformation(llmResponse: list[str]):

    llmResponse = [line for line in llmResponse if not line.isspace()]
    programcode = "\n".join(llmResponse)
    programcode = extract_code(programcode)
    programcode = programcode.split("\n")

    functionName = programcode[0].split()[1].upper().removesuffix(".")
    importParameters, exportParameters = parseAbapFunctionInterface(programcode)

    programcode = [line for line in programcode if not line.startswith("*")]
    programcode = (
        programcode[1:] if programcode[0].startswith("FUNCTION") else programcode
    )
    programcode = (
        programcode[:-1] if programcode[-1].startswith("ENDFUNCTION.") else programcode
    )

    return functionName, importParameters, exportParameters, programcode


def askLLMForPromptsSingleThread(
    savefilename: str,
    threadId: int,
    llmModel: str,
    sleepTime: float,
    requestList: list[LLMRequest],
    q: queue.Queue,
):
    if savefilename.endswith(".json"):
        savefilename = savefilename.removesuffix(".json")
    progressCounter = 0

    saveDict = {}

    for request in requestList:
        llm = Ollama(systemMessage=request.systemPrompt, model=llmModel)
        # llm = ChatGPT(systemMessage=request.systemPrompt, temperature=request.temperature, model=llmModel)

        if request.systemPrompt not in saveDict:
            saveDict[request.systemPrompt] = {}
        if str(request.temperature) not in saveDict[request.systemPrompt]:
            saveDict[request.systemPrompt][str(request.temperature)] = {}
        if (
            request.promptOption
            not in saveDict[request.systemPrompt][str(request.temperature)]
        ):
            saveDict[request.systemPrompt][str(request.temperature)][
                request.promptOption
            ] = {}
        if (
            str(request.promptNumber)
            not in saveDict[request.systemPrompt][str(request.temperature)][
                request.promptOption
            ]
        ):
            saveDict[request.systemPrompt][str(request.temperature)][
                request.promptOption
            ][str(request.promptNumber)] = {}
        if (
            "attempts"
            not in saveDict[request.systemPrompt][str(request.temperature)][
                request.promptOption
            ][str(request.promptNumber)]
        ):
            saveDict[request.systemPrompt][str(request.temperature)][
                request.promptOption
            ][str(request.promptNumber)]["attempts"] = []

        prompt = getPrompt(request.promptOption, request.promptNumber)
        saveDict[request.systemPrompt][str(request.temperature)][request.promptOption][
            str(request.promptNumber)
        ]["prompt"] = prompt

        saveTo = {}
        llmResponse = None
        tries = 0
        while llmResponse == None and tries < 10:
            try:
                llmResponse = llm.askWithoutContext(prompt).split("\n")
            except Exception:
                llmResponse = None
                print(
                    "Thread", threadId, "crashed", tries, "times, retrying in 5 seconds"
                )
                tries += 1
                time.sleep(sleepTime)
        saveTo["llmResponse"] = llmResponse
        saveDict[request.systemPrompt][str(request.temperature)][request.promptOption][
            str(request.promptNumber)
        ]["attempts"].append(saveTo)
        progressCounter += 1
        print(
            "Thread "
            + str(threadId)
            + " Progress "
            + str(progressCounter)
            + "/"
            + str(len(requestList))
        )

        if progressCounter.__mod__(50) == 0:
            with open(
                "backup/"
                + savefilename
                + " thread"
                + str(threadId)
                + " progress"
                + str(progressCounter)
                + ".json",
                "w",
            ) as file:
                file.write(json.dumps(saveDict, indent=4))

        time.sleep(5)
    with open(
        "backup/"
        + savefilename
        + " thread"
        + str(threadId)
        + " progress"
        + str(progressCounter)
        + ".json",
        "w",
    ) as file:
        file.write(json.dumps(saveDict, indent=4))

    q.put(saveDict)


def askLLMForPromptsMultithreaded(
    savefilename: str,
    systemPrompts,
    promptOptions,
    temperatures,
    llmModel: str,
    sleepTime: float,
    promptNumbers: list[int],
    numRepeats: int,
    numThreads: int,
):
    saveDict = {}
    requests: list[LLMRequest] = []

    # initialize dict
    for systemPrompt in systemPrompts:
        saveDict[systemPrompt] = {}
        for temperature in temperatures:
            saveDict[systemPrompt][str(temperature)] = {}
            for promptOption in promptOptions:
                saveDict[systemPrompt][str(temperature)][promptOption] = {}
                for promptNumber in promptNumbers:
                    saveDict[systemPrompt][str(temperature)][promptOption][
                        str(promptNumber)
                    ] = {}
                    prompt = getPrompt(promptOption, promptNumber)
                    saveDict[systemPrompt][str(temperature)][promptOption][
                        str(promptNumber)
                    ]["prompt"] = prompt
                    saveDict[systemPrompt][str(temperature)][promptOption][
                        str(promptNumber)
                    ]["attempts"] = []
                    for repeat in range(0, numRepeats):
                        requests.append(
                            LLMRequest(
                                systemPrompt, promptOption, temperature, promptNumber
                            )
                        )

    # calculate which thread does what
    numAllIterations = (
        len(systemPrompts)
        * len(promptOptions)
        * len(temperatures)
        * len(promptNumbers)
        * numRepeats
    )
    numThreads = min(numThreads, numAllIterations)
    print("distributing", numAllIterations, "prompts to", numThreads, "threads")
    threadLists = [list(i) for i in more_itertools.divide(numThreads, requests)]

    # print(len(threadLists))
    # for thread in threadLists:
    #     print(thread)
    # for item in thread:
    #     print(item)

    # start threads
    q = queue.Queue()
    pool = []
    for index, requestList in enumerate(threadLists):
        print("creating thread " + str(index))
        t = threading.Thread(
            target=askLLMForPromptsSingleThread,
            args=(savefilename, index, llmModel, sleepTime, requestList, q),
        )
        t.start()
        pool.append(t)

    # save thread data
    for index, t in enumerate(pool):
        t.join()
        response = q.get()
        responseSystemPrompts = list(response.keys())
        for responseSystemPrompt in responseSystemPrompts:
            responseTemperatures = list(response[responseSystemPrompt].keys())
            for responseTemperature in responseTemperatures:
                responsePromptOptions = list(
                    response[responseSystemPrompt][responseTemperature].keys()
                )
                for responsePromptOption in responsePromptOptions:
                    responsePromptNums = list(
                        response[responseSystemPrompt][responseTemperature][
                            responsePromptOption
                        ].keys()
                    )
                    for responsePromptNum in responsePromptNums:
                        responseAttempts = response[responseSystemPrompt][
                            responseTemperature
                        ][responsePromptOption][responsePromptNum]["attempts"]
                        # print(responseAttempts)
                        for attempt in responseAttempts:
                            saveDict[responseSystemPrompt][str(responseTemperature)][
                                responsePromptOption
                            ][responsePromptNum]["attempts"].append(attempt)

    with open(savefilename, "w") as file:
        file.write(json.dumps(saveDict, indent=4))


def runFunction(
    prompts,
    systemPrompt,
    temperature,
    promptOption,
    promptNumber,
    attemptNr,
    progressCounter,
    connection_params,
    savefilename,
    functionpool: str
):
    currentAttempt = prompts[systemPrompt][temperature][promptOption][promptNumber][
        "attempts"
    ][attemptNr]
    print(
        "Progress: ",
        str(progressCounter),
        "/",
        str(
            len(prompts)
            * len(prompts[systemPrompt])
            * len(prompts[systemPrompt][temperature])
            * len(prompts[systemPrompt][temperature][promptOption])
            * len(
                prompts[systemPrompt][temperature][promptOption][promptNumber][
                    "attempts"
                ]
            )
        ),
    )
    if (
        "functionCreated" not in currentAttempt
        or currentAttempt["functionCreated"] == "CommunicationErrorRFC_CLOSED"
        or currentAttempt["functionCreated"].startswith(
            "FunctionCreate: ApplicationError 5 (rc=5): key=FUNCTION_ALREADY_EXISTS"
        )
        or currentAttempt["functionCreated"].startswith(
            "FunctionCreate: ApplicationError 5 (rc=5): key=TOO_MANY_FUNCTIONS"
        )
        or currentAttempt["functionCreated"].startswith(
            "FunctionCreate: RuntimeError DBSQL_SQL_ERROR"
        )
    ):
        try:
            functionName, importParameters, exportParameters, programcode = (
                extractAbapFunctionInformation(currentAttempt["llmResponse"])
            )
        except Exception as e:
            functionName, importParameters, exportParameters, programcode = (
                "defaultFunctionName",
                "",
                "",
                "",
            )
            # print(e)
        
        
        if re.search("[zZ]_[a-zA-Z0-9_]{0,28}", functionName) == None:
            currentAttempt["functionnameModified"] = True

        time_millis = round(time.time() * 1000)
        functionName = f"Z_{time_millis}"

        currentAttempt["functionname"] = functionName
        currentAttempt["importParameters"] = importParameters
        currentAttempt["exportParameters"] = exportParameters
        currentAttempt["programcode"] = programcode

        testModule = importlib.import_module("adjustedTests.test" + str(promptNumber))

        if currentAttempt["functionname"].lower().startswith("z"):
            with pyrfc.Connection(**connection_params) as conn:
                functionCreated = pythonAbapInterface.createFunctionModule(
                    conn,
                    currentAttempt["functionname"],
                    currentAttempt["importParameters"],
                    currentAttempt["exportParameters"],
                    currentAttempt["programcode"],
                    functionpool
                )

                currentAttempt["functionCreated"] = functionCreated
                time.sleep(2)

                passed, failed = testModule.check(
                    Test(
                        conn,
                        currentAttempt["functionCreated"],
                        currentAttempt["functionname"],
                        currentAttempt["importParameters"],
                        currentAttempt["exportParameters"],
                        currentAttempt,
                    ).callFunction
                )
                time.sleep(2)

        else:
            currentAttempt["functionCreated"] = "function name does not start with z"
            passed, failed = 0, testModule.num_tests()

        currentAttempt["passed"] = passed
        currentAttempt["failed"] = failed
        currentAttempt["tests"] = passed + failed
        print(
            f"Prompt {promptNumber} Attempt {attemptNr}: {passed} out of {passed+failed} unit-tests were successful"
        )

        with pyrfc.Connection(**connection_params) as conn:
            pythonAbapInterface.deleteFunctionModule(
                conn, currentAttempt["functionname"]
            )

    return currentAttempt


def runSavedFunctions(loadfilename: str, savefilename: str, functionpool: str, start_at: int = 0):
    if savefilename.endswith(".json"):
        savefilename = savefilename.removesuffix(".json")
    with open(loadfilename) as file:
        prompts = json.loads(file.read())

    with open("saplogonLoginDetails - local.json", "r") as file:
        connection_params = json.loads(file.read())

    progressCounter = 0

    for systemPromptIndex, systemPrompt in enumerate(prompts):
        for tempIndex, temperature in enumerate(prompts[systemPrompt]):
            for promptOptionIndex, promptOption in enumerate(
                prompts[systemPrompt][temperature]
            ):
                for promptNumber in prompts[systemPrompt][temperature][promptOption]:
                    for attemptNr, attempt in enumerate(
                        prompts[systemPrompt][temperature][promptOption][promptNumber][
                            "attempts"
                        ]
                    ):
                        progressCounter += 1
                        if start_at > progressCounter:
                            continue

                        currentAttempt = runFunction(
                            prompts,
                            systemPrompt,
                            temperature,
                            promptOption,
                            promptNumber,
                            attemptNr,
                            progressCounter,
                            connection_params,
                            savefilename,
                            functionpool
                        )

                        if currentAttempt["functionCreated"] == "CommunicationErrorRFC_CLOSED" \
                            or currentAttempt["functionCreated"].startswith("FunctionCreate: ApplicationError 5 (rc=5): key=FUNCTION_ALREADY_EXISTS") \
                            or currentAttempt["functionCreated"] == "FunctionCreate: RuntimeError DBSQL_SQL_ERROR":  # type: ignore
                            print(f"died again. saving progress {progressCounter}. Reason: {currentAttempt["functionCreated"]}")




                            with open(
                                "backup/"
                                + savefilename
                                + " progress"
                                + str(progressCounter)
                                + ".json",
                                "w",
                            ) as file:
                                file.write(json.dumps(prompts, indent=4))
                                sys.exit()

                        prompts[systemPrompt][str(temperature)][promptOption][
                            str(promptNumber)
                        ]["attempts"][attemptNr] = currentAttempt

                        if (progressCounter.__mod__(100) == 0) and savefilename != None:
                            with open(
                                "backup/"
                                + savefilename
                                + " progress"
                                + str(progressCounter)
                                + ".json",
                                "w",
                            ) as file:
                                file.write(json.dumps(prompts, indent=4))

    if savefilename != None:
        with open(savefilename + ".json", "w") as file:
            file.write(json.dumps(prompts, indent=4))


def runTestFunction():
    with open("saplogonLoginDetails - local.json", "r") as file:
        connection_params = json.loads(file.read())

    with pyrfc.Connection(**connection_params) as conn:
        res = pythonAbapInterface.callFunctionModule(
            conn, "Z_TEST", {"NUM_ONE": 12, "NUM_TWO": 3}
        )
        print(res)


def find_left_off_position(latest_backup_filename: str) -> int:
    progressCounter = 0
    with open(latest_backup_filename, "r") as file:
        prompts = json.loads(file.read())
        for systemPromptIndex, systemPrompt in enumerate(prompts):
            for tempIndex, temperature in enumerate(prompts[systemPrompt]):
                for promptOptionIndex, promptOption in enumerate(
                    prompts[systemPrompt][temperature]
                ):
                    for promptNumber in prompts[systemPrompt][temperature][
                        promptOption
                    ]:
                        for attemptNr, attempt in enumerate(
                            prompts[systemPrompt][temperature][promptOption][
                                promptNumber
                            ]["attempts"]
                        ):
                            progressCounter += 1
                            currentAttempt = prompts[systemPrompt][temperature][
                                promptOption
                            ][promptNumber]["attempts"][attemptNr]

                            if (
                                not "functionCreated" in currentAttempt
                                or currentAttempt["functionCreated"]
                                == "CommunicationErrorRFC_CLOSED"
                                or currentAttempt["functionCreated"].startswith(
                                    "FunctionCreate: ApplicationError 5 (rc=5): key=FUNCTION_ALREADY_EXISTS"
                                ) or currentAttempt["functionCreated"] == "FunctionCreate: RuntimeError DBSQL_SQL_ERROR"):
                                return progressCounter
    return progressCounter
