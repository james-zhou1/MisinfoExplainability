import json
import openai
import pandas as pd

def testExplanationFormatValid(explanation):
    return '. ' in explanation

def testClassificationValid(classification):
    valid_classifications = ["true", "mostly-true", "half-true", "mostly-false", "false", "pants-fire"]
    if classification.endswith('.'):
        classification = classification[:-1]
    return classification in valid_classifications, f"Invalid classification: {classification}"

with open('key.env', 'r') as file:
    API_KEY = file.read().strip().split('=')[1]
openai.api_key = API_KEY

with open('LIAR/LIAR_val.jsonl', 'r') as file:
    lines = file.readlines()[:20]
    print("The size of LIAR dataset is:", len(lines))

    dataset = []
    for line in lines:
        entry = json.loads(line)
        text = entry['text']

        while (True):
            explanation = openai.Completion.create(
                engine="text-davinci-003",
                prompt=f"In the first sentence, classify the following statement as true, mostly-true, half-true, mostly-false, false, or pants-fire, without additional words. After that, explain your reasoning for your classification: {text}.",
                max_tokens=110,
                n=1,
            ).choices[0].text.strip()
            
            if not testExplanationFormatValid(explanation):
                print("No period to split the query, trying again...")
                continue

            classification, explanation = explanation.split('. ', 1)
            classification = classification.lower()

            if not testClassificationValid(classification):
                print("Not a valid classification, trying again...")
                continue

            print("Text:", text)
            print("Classification:", classification)
            print("Explanation:", explanation)
                
            dataset.append({'classification': classification, 'text': text, 'explanation': explanation})
            break

    with open('dataset.json', 'w') as json_file:
        json.dump(dataset, json_file, indent=4)

    print("Dataset created as a JSON file: dataset.json")


