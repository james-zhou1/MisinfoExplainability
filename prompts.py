import openai

with open('key.env', 'r') as file:
    API_KEY = file.read().strip().split('=')[1]

openai.api_key = API_KEY

import pandas as pd

data = pd.read_csv('CT-FAN-22\English_data_test_release_with_rating.csv')
prompts = data['text'].tolist()[:5]

# print(prompts[0])

tones = ["neutral", "emotional", "authoritative", "casual"]
scores = {}

for prompt in prompts:
  for tone in tones:
    response = openai.Completion.create(
      engine="text-davinci-003",
      prompt=f"Classify the following statement as true, mostly-true, half-true, mostly-false, false, or pants-fire. Then, explain your classification in a {tone} tone: {prompt}.",
      max_tokens=60
    ).choices[0].text.strip()

    score = openai.Completion.create(
      engine="text-davinci-003",
      prompt=f"Provide a score for this statement from 0 to 100, where 0 represents a poor explanation and 100 represents a good explanation. Do not provide any explanations, only respond with the numerical score: {response}",
      max_tokens=60 
    ).choices[0].text.strip()

    print(tone, score)
    print(response)
    print("\n")

    scores.setdefault(tone, []).append(score)

print(scores.items())

for tone, score in scores.items():
  score = sum(int(s) for s in score) / len(score)
  print("Average score of:", tone, score)