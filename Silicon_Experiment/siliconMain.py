from siliconModels import explainers, evaluators, labeler
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

dataset = []
file = open('merged.jsonl', 'r')
lines = file.readlines()[:2]

for line in lines:
    input = json.loads(line)
    text = input['statement']
    possibility_label = input['possibility_label']
    output = []
    print(f"Next Line")


    ### Hacky classifier model start
    _, _, given_label, _ = labeler.get_response(text)
    actual_label = input['label']

    print("Given_label", given_label)
    print("actual label", actual_label)

    match = (given_label.lower() == actual_label.lower())

### Hacky classifier model end
    answer = input['gpt-answer']
    uncertainty = abs(50 - int(answer))

    for explainer in explainers:
        explainer_config, explainer_prompt, explanation, explainer_cost = explainer.get_response(text)
        for evaluator in evaluators:
            while True:
                evaluator_config, evaluator_prompt, evaluation, evaluator_cost = evaluator.get_response(explanation)
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
                'uncertainty': uncertainty
            }
            dataset.append(data_entry)
            print(explainer_config, evaluator_config)

with open('scores.json', 'w') as json_file: json.dump(dataset, json_file, indent=4)

def displayResults(attribute, possibility_label=None, subplot=0):
    matrix_sum = defaultdict(lambda: defaultdict(int))
    matrix_count = defaultdict(lambda: defaultdict(int))

    filtered_dataset = [data_entry for data_entry in dataset if data_entry['possibility_label'] == possibility_label]
    
    if not filtered_dataset:
        print(f"No data entries for label {possibility_label}")
        return
    print(f"{len(filtered_dataset)} data entries for label {possibility_label}")
    
    for data_entry in filtered_dataset:
        matrix_sum[data_entry['evaluator_config']][data_entry['explainer_config']] += float(data_entry[attribute])
        matrix_count[data_entry['evaluator_config']][data_entry['explainer_config']] += 1
    average_matrix = matrix_sum.copy()
    for evaluator in average_matrix:
        for explainer in average_matrix[evaluator]:
            average_matrix[evaluator][explainer] /= matrix_count[evaluator][explainer]

    df = pd.DataFrame(average_matrix)
    df = df.apply(pd.to_numeric, errors='coerce')

    plt.subplot(3, 3, subplot)
    sns.heatmap(df, annot=True, cmap='YlGnBu')

    plt.title(f'{attribute} Plot for {possibility_label}')
    plt.xlabel('Evaluator')
    plt.ylabel('Explainer')

metrics = ['evaluation', 'explainer_cost', 'evaluator_cost']
difficulties = ['impossible', 'hard', 'possible']


for metric in metrics:
    subplot = 1
    for difficulty in difficulties:
        displayResults(metric, difficulty[0], subplot)   #   Note that the label is represented by only the first character.
        subplot += 1
    plt.tight_layout()
    plt.show()





import matplotlib.pyplot as plt

# Extract uncertainty and evaluations from the dataset
uncertainties = [entry['uncertainty'] for entry in dataset]
evaluations = [entry['evaluation'] for entry in dataset]

# Create the scatter plot
plt.scatter(uncertainties, evaluations)
plt.xlabel('Uncertainty')
plt.ylabel('Evaluation')
plt.show()



import matplotlib.pyplot as plt

# Extract evaluations and matches from the dataset
import matplotlib.pyplot as plt
import numpy as np

# Separate evaluations into matches and non-matches
match_evaluations = [float(entry['evaluation']) for entry in dataset if entry['match']]
non_match_evaluations = [float(entry['evaluation']) for entry in dataset if not entry['match']]

# Calculate average evaluation for matches and non-matches
avg_match_evaluation = np.mean(match_evaluations) if match_evaluations else 0
avg_non_match_evaluation = np.mean(non_match_evaluations) if non_match_evaluations else 0

# Create the bar plot
plt.bar(['Match', 'No Match'], [avg_match_evaluation, avg_non_match_evaluation])
plt.xlabel('Match Status')
plt.ylabel('Average Evaluation Score')
plt.show()



from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# Create a dictionary where the keys are the explainer parameters and the values are lists of evaluation scores
explainer_evaluations = defaultdict(list)

for entry in dataset:
    explainer_evaluations[entry['explainer_config']].append(entry['evaluation'])

# Calculate the average evaluation score for each explainer
explainer_averages = {explainer: np.mean(scores) for explainer, scores in explainer_evaluations.items()}

# Create the bar plot
plt.bar(explainer_averages.keys(), explainer_averages.values())
plt.xlabel('Explainer')
plt.ylabel('Average Evaluation Score')
plt.show()