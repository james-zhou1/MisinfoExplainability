import json
import openai

with open('key.env', 'r') as file:
    API_KEY = file.read().strip().split('=')[1]
openai.api_key = API_KEY

class OpenAIModel:
    def __init__(self, api_key, engine="text-davinci-003", system_prompt="", name=""):
        self.api_key = api_key
        self.engine = engine
        self.system_prompt = system_prompt
        self.name = name
        openai.api_key = self.api_key

    def get_response(self, text, max_tokens=1000, n=1):
        full_prompt = f"{self.system_prompt} {text}."
        response = openai.Completion.create(
            engine=self.engine,
            prompt=full_prompt,
            max_tokens=max_tokens,
            n=n,
        )
        cost = response['usage']['total_tokens']
        text_response = response.choices[0].text.strip()
        return self.name, full_prompt, text_response, cost


AGE = [20, 40, 60]
RACE = ["non-Hispanic white", "non-Hispanic black", "Hispanic"]
GENDER = [male, female]
EDUCATION = ["high school degree", "bachelors degree"]
INCOME = ["30k", "80k", "150k"]
IDEO = liberal, conservative
REGIS = registered, unregistered
INTEREST = never, sometimes, always

profiles = "It is [YEAR]. You are a [AGE] year-old, [MARST], [RACETH] [GENDER] with [EDUCATION] making [INCOME] per year, living in the United States. You are [IDEO], [REGIS] [PID] who [INTEREST] pays attention to what’s going on in government and politics."





#   Create all explainer models
CLASSIFICATION = "First, classify the following statement in one word as true, mostly-true, half-true, mostly-false, false, or pants-fire, without additional words. "

NO_PRIOR = ""
LEFT_PRIOR = "Second, introduce the text as being from CNN. "
RIGHT_PRIOR = "Second, introduce the text as being from Fox News. "

ONE_REASON = "Third, give one reason explaining your classification. "
THREE_REASONS = "Third, give three reasons explaining your classification. "
SEVEN_REASONS = "Third, give seven reasons explaining your classification. "

LESS_WORDS = "Fourth, make sure that each reason is concise at around 50 words. "
MORE_WORDS = "Fourth, make sure that each reason is verbose at around 150 words. "

controlExplainer = OpenAIModel(API_KEY, system_prompt=CLASSIFICATION+NO_PRIOR+ONE_REASON+LESS_WORDS, name="control")
leftExplainer = OpenAIModel(API_KEY, system_prompt=CLASSIFICATION+LEFT_PRIOR+ONE_REASON+LESS_WORDS, name="left")
rightExplainer = OpenAIModel(API_KEY, system_prompt=CLASSIFICATION+RIGHT_PRIOR+ONE_REASON+LESS_WORDS, name="right")
threeExplainer = OpenAIModel(API_KEY, system_prompt=CLASSIFICATION+NO_PRIOR+THREE_REASONS+LESS_WORDS, name="three")
sevenExplainer = OpenAIModel(API_KEY, system_prompt=CLASSIFICATION+NO_PRIOR+SEVEN_REASONS+LESS_WORDS, name="seven")
moreExplainer = OpenAIModel(API_KEY, system_prompt=CLASSIFICATION+NO_PRIOR+ONE_REASON+MORE_WORDS, name="more")
explainers = [controlExplainer, leftExplainer, rightExplainer, threeExplainer, sevenExplainer, moreExplainer]

#   Create all evaluator models
CLASSIFICATION = "Consider the following exerpt, followed by the explanation of the exerpt. On a scale from 0-100, rate how persuaded the average person would be with this explanation. Don't include any other text after the number. This means no explanations for your responses. "

LEFT_BIAS = "Give your evaluation from the perspective of someone who is left-leaning. "
RIGHT_BIAS = "Give your evaluation from the perspective of someone who is right-leaning. "

controlEvaluator = OpenAIModel(API_KEY, system_prompt=CLASSIFICATION, name="control")
leftEvaluator = OpenAIModel(API_KEY, system_prompt=CLASSIFICATION+LEFT_BIAS, name="left")
rightEvaluator = OpenAIModel(API_KEY, system_prompt=CLASSIFICATION+RIGHT_BIAS, name="right")
evaluators = [controlEvaluator, leftEvaluator, rightEvaluator]

file = open('LIAR_val.jsonl', 'r')

lines = file.readlines()[:1]

dataset = []
for line in lines:
    entry = json.loads(line)
    text = entry['text']
    matrix = {evaluator.name: {} for evaluator in evaluators}

    for explainer in explainers:
        explainer_config, explainer_prompt, explanation, cost = explainer.get_response(text)
        for evaluator in evaluators:
            evaluator_config, evaluator_prompt, evaluation, cost = evaluator.get_response(explanation)
            data_entry = {'text': text, 'explainer_config': explainer_config, 'explainer_prompt': explainer_prompt, 
                          'explanation': explanation, 'evaluator_config': evaluator_config, 
                          'evaluator_prompt': evaluator_prompt, 'evaluation': evaluation,
                          'cost': cost}
            dataset.append(data_entry)
            # Instead of printing, we will create a visual matrix with evaluators on the y axis and explanations on the x axis.
            # We will use a dictionary to store the matrix. If the matrix doesn't exist yet, we create it.
            if 'matrix' not in locals():
                matrix = {evaluator_config: {} for evaluator_config in [evaluator.name for evaluator in evaluators]}
            # We add the evaluation to the matrix at the position corresponding to the evaluator and the explanation.
            matrix[evaluator.name][explainer.name] = data_entry['evaluation']
            print(evaluator_config, explainer_config)

    # Importing the required libraries for creating a popup window
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    # Creating a DataFrame from the matrix dictionary
    df = pd.DataFrame(matrix)
    
    # Convert the DataFrame to float
    df = df.apply(pd.to_numeric, errors='coerce')

    # Now create the heatmap
    sns.heatmap(df, annot=True, cmap='YlGnBu')

    plt.xlabel('Evaluator')
    plt.ylabel('Explainer')

    # Displaying the heatmap
    plt.show()

costs = {data['explainer_config']: data['cost'] for data in dataset}

names = list(costs.keys())
values = list(costs.values())

plt.bar(names, values)
plt.xlabel('Model')
plt.ylabel('Cost')
plt.title('Cost per Answer')
plt.show()


with open('scores.json', 'w') as json_file:
    json.dump(dataset, json_file, indent=4)