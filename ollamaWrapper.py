from time import sleep
import ollama


retryTime = 5

class Ollama:
    def __init__(self, systemMessage, model: str) -> None: #temperature=0.8, 
        self.context = []
        self.systemMessage = systemMessage
        # self.temperature = temperature
        self.model = model

    def _runQuery(self, messages):
        
        attempts = 0
        completion = ""
        while completion == "" and attempts <= 10:
            try:
                completion = ollama.chat(
                    model=self.model,
                    # temperature=self.temperature, TODO
                    messages=messages)
                    
            except (ollama.RequestError, ollama.ResponseError) as e:
                attempts += 1
                print(e)
                print(f"LLM APIError: trying again in {retryTime} seconds...")
                sleep(retryTime)
            
            if attempts > 10:
                return "LLM APIError"
        return completion["message"]["content"] # type: ignore
    
    def askWithoutContext(self, userMessage):
        messages = []
        if self.systemMessage != "":
            messages.append({"role": "system", "content": self.systemMessage})
        if isinstance(userMessage, str):
            messages.append({"role": "user", "content": userMessage})
        if isinstance(userMessage, list):
            for message in userMessage:
                messages.append({"role": "user", "content": message})

        response = self._runQuery(messages)
        return str(response)
    
    def askWithContext(self, userMessage):
        messages = []
        if self.systemMessage != "":
            messages.append({"role": "system", "content": self.systemMessage})
        for message in self.context:
            messages.append(message)
        messages.append({"role": "user", "content": userMessage})
        
        response = self._runQuery(messages)
        self.context.append({"role": "assistant", "content": response})
        return str(response)
    
    def listModels(self):
        models = ollama.list()
        if isinstance(models, dict):
            return [i["id"] for i in models["data"]]
        
        