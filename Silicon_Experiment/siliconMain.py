from siliconModels import explainers, evaluators, labeler
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

dataset = []
file = open('merged.jsonl', 'r')
lines = file.readlines()[:70]

count = 1
for line in lines:
    print(f"Next Line {count}")
    count += 1

    input = json.loads(line)
    text = input['statement']

    possibility_label = input['possibility_label']

    _, _, given_label, _ = labeler.get_response(text)
    actual_label = input['label']
    match = (given_label.lower() == actual_label.lower())

    answer = input['gpt-answer']
    confidence = abs(50 - int(answer))

    for explainer in explainers:
        explainer_config, explainer_prompt, explanation, explainer_cost = explainer.get_response(f"Your confidence in your explanation is going to be {confidence}." + text)
        for evaluator in evaluators:
            print(f"Next Line {count}")
            while True:
                evaluator_config, evaluator_prompt, evaluation, evaluator_cost = evaluator.get_response(f"On a scale from 0-100, rate how persuasive this explanation is for the purposes of convincing someone that the statement is definitely True or definitely False, as opposed to uncertain." + explanation)
                try:
                    int_evaluation = int(evaluation)
                    break
                except ValueError:
                    print("Explanation:", explanation)
                    print("Evaluation:", evaluation)
                    print(f"Cannot convert evaluation to int: {evaluation}. Retrying...")
            data_entry = {
                'text': text, 
                'explainer_config': explainer_config, 
                'explainer_prompt': explainer_prompt, 
                'explanation': explanation,
                'explainer_cost': explainer_cost,
                'evaluator_config': evaluator_config, 
                'evaluator_prompt': evaluator_prompt, 
                'evaluation': evaluation,
                'evaluator_cost': evaluator_cost,
                'possibility_label': possibility_label,
                'match': match,
                'confidence': confidence
            }
            dataset.append(data_entry)
            print(explainer_config, evaluator_config)

with open('scores.json', 'w') as json_file: json.dump(dataset, json_file, indent=4)