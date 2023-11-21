import json

# Load the first dataset
with open('LIAR-New.jsonl', 'r') as f:
    liar_new = [json.loads(line) for line in f]

# Load the second dataset
with open('LIAR_new_gpt-4_repeat0_temp0.5 (4).jsonl', 'r') as f:
    liar_new_gpt3 = [json.loads(line) for line in f]

# Convert the second dataset to a dictionary for easy lookup
liar_new_gpt3_dict = {item['example_id']: item for item in liar_new_gpt3}

# Merge the datasets
merged = []
for item in liar_new:
    example_id = item['example_id']
    if example_id in liar_new_gpt3_dict:
        # Merge the dictionaries
        merged_item = {**item, **liar_new_gpt3_dict[example_id]}
        merged.append(merged_item)

# Save the merged dataset
with open('merged.jsonl', 'w') as f:
    for item in merged:
        f.write(json.dumps(item) + '\n')