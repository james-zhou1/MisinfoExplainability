CLASSIFICATION = "First, classify the following statement in one word as true, mostly-true, half-true, mostly-false, false, or pants-fire, without additional words. "

LEFT_PRIOR = "Second, introduce yourself as being a political analyst at CNN. "
RIGHT_PRIOR = "Second, introduce yourself as being a political analyst at Fox News. "

ONE_REASON = "Third, give one reason explaining your classification. "
THREE_REASONS = "Third, give three reasons explaining your classification. "
SEVEN_REASONS = "Third, give seven reasons explaining your classification. "

LESS_WORDS = "Fourth, make sure that each reason is concise at around 50 words. "
MORE_WORDS = "Fourth, make sure that each reason is verbose at around 150 words. "

prompts = []

combinations = []
for prior in [LEFT_PRIOR, RIGHT_PRIOR]:
    for reason in [ONE_REASON, THREE_REASONS, SEVEN_REASONS]:
        for words in [LESS_WORDS, MORE_WORDS]:
            combinations.append((prior, reason, words))

for combination in combinations:
    prompt = {
        "engine": "text-davinci-003",
        "prompt": "Follow the instructions in order. " + str(CLASSIFICATION + combination[0] + combination[1] + combination[2]),
        "max_tokens": 300,
        "n": 1,
    }
    prompts.append(prompt)