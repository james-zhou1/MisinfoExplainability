import openai

with open('key.env', 'r') as file:
    API_KEY = file.read().strip().split('=')[1]
openai.api_key = API_KEY

class OpenAIModel:
    def __init__(self, api_key, engine="text-davinci-003", system_prompt="Act as a neutral political actor."):
        self.api_key = api_key
        self.engine = engine
        self.system_prompt = system_prompt
        openai.api_key = self.api_key

    def get_response(self, text, max_tokens=110, n=1):
        full_prompt = f"{self.system_prompt} {text}."
        response = openai.Completion.create(
            engine=self.engine,
            prompt=full_prompt,
            max_tokens=max_tokens,
            n=n,
        ).choices[0].text.strip()
        return response
    
LEFT = "Output an integer from 0 to 10 describing the likelihood that a US Democrat would agree with the following statement. Take into account biases that such a person might have. After giving your answer, give an explanation."
RIGHT = "Output an integer from 0 to 10 describing the likelihood that a US Republican would agree with the following statement. Take into account biases that such a person might have. After giving your answer, give an explanation."

leftModel = OpenAIModel(API_KEY, system_prompt=LEFT)
rightModel = OpenAIModel(API_KEY, system_prompt=RIGHT)

def prompt(text):
    print(text)
    print(leftModel.get_response(text))
    print(rightModel.get_response(text))
prompt("Climate change is a hoax")
prompt("Trump was a good president")
prompt("Biden was a good president")
prompt("The elections were stolen")
prompt("The COVID vaccine doesn't work")
prompt("Medicare for All has no economic implications")
