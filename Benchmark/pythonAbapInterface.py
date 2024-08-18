import pyrfc
def createFunctionModule(conn, functionName, importParameters, exportParameters, programcode, function_pool):
    try: 
        result = conn.call('RS_FUNCTIONMODULE_INSERT',
                        FUNCNAME=functionName,
                        FUNCTION_POOL=function_pool,
                        SHORT_TEXT="",
                        REMOTE_CALL="X",
                        SOURCE=programcode,
                        IMPORT_PARAMETER=importParameters,
                        EXPORT_PARAMETER=exportParameters)
        return "success"
    except pyrfc._exception.ABAPApplicationError as e: # type: ignore
        return "FunctionCreate: ApplicationError " + str(e)
        pass
    except pyrfc._exception.ABAPRuntimeError as e: # type: ignore
        return "FunctionCreate: RuntimeError " + str(e.key)
        pass
    except pyrfc._exception.LogonError as e: # type: ignore
        return "FunctionCreate: LogonError " + str(e.key)
        pass
    except pyrfc._exception.CommunicationError as e: # type: ignore
        return "CommunicationError" + str(e.key)
        pass
    except Exception as e:
        return str(e)

def callFunctionModule(conn, functionName, callParams) -> dict:
    try:
        result = conn.call(functionName, options={"timeout": 10}, **callParams)
        return result
    except pyrfc._exception.ABAPApplicationError as e: # type: ignore
        result = "FunctionCall: ApplicationError " + str(e.key)
    except pyrfc._exception.ABAPRuntimeError as e: # type: ignore
        result = "FunctionCall: RuntimeError " + str(e)
    except pyrfc._exception.LogonError as e: # type: ignore
        result = "FunctionCall: LogonError " + str(e.key)
    except pyrfc._exception.CommunicationError as e: # type: ignore
        result = "FunctionCall: CommunicationError " + str(e.key)
    except TypeError as e:
        result = "FunctionCall: TypeError: " + str(callParams)
    except Exception as e:
        result = repr(e)
    return {"exception": result}

def deleteFunctionModule(conn, functionName):
    res = conn.call("Z_FUNCTION_DELETE", FUNCNAME=functionName)