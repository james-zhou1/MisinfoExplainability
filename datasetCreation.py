import json
import openai
from datasetTests import testExplanationFormatValid, testClassificationValid, testClassificationCorrect
from datasetPrompts import prompts

API_KEY = open('key.env', 'r').read().strip().split('=')[1]
openai.api_key = API_KEY

file = open('LIAR_val.jsonl', 'r')

lines = file.readlines()[:1]
print("The size of LIAR dataset is:", len(lines))

truthfulness_mapping = {
        0: "pants-fire",
        1: "false",
        2: "mostly-false",
        3: "half-true",
        4: "mostly-true",
        5: "true",
    }

def generateExplanation(entry):
    text = entry['text']
    label = truthfulness_mapping[entry['label']].lower()
    for prompt in prompts:
        prompt["prompt"] = f"The accompanying text is {label} " + prompt["prompt"] + f"Here is the text: {text}"
        print("==============================================")
        print("The prompt is:", prompt["prompt"])
        while (True):
            explanation = openai.Completion.create(**prompt).choices[0].text.strip()
            
            if not testExplanationFormatValid(explanation): continue

            classification, explanation = explanation.split('. ', 1)
            classification = classification.lower()
            if not testClassificationValid(classification): continue
            if not testClassificationCorrect(classification, label): continue
            dataset.append({'prompt': prompt, 'classification': classification, 'text': text, 'explanation': explanation})
            print("The explanation is:", explanation)
            break

dataset = []
for line in lines:
    entry = json.loads(line)
    
    generateExplanation(entry)


with open('dataset.json', 'w') as json_file:
    json.dump(dataset, json_file, indent=4)

print("Dataset created as a JSON file: dataset.json")


file.close()