CLASSIFICATION = "First, classify the following statement in one word as true, mostly-true, half-true, mostly-false, false, or pants-fire, without additional words. "

NO_PRIOR = ""
LEFT_PRIOR = "Second, introduce the text as being from CNN. "
RIGHT_PRIOR = "Second, introduce the text as being from Fox News. "

ONE_REASON = "Third, give one reason explaining your classification. "
THREE_REASONS = "Third, give three reasons explaining your classification. "
SEVEN_REASONS = "Third, give seven reasons explaining your classification. "

LESS_WORDS = "Fourth, make sure that each reason is concise at around 50 words. "
MORE_WORDS = "Fourth, make sure that each reason is verbose at around 150 words. "

prompts = []

combinations = []
for prior in [NO_PRIOR, LEFT_PRIOR, RIGHT_PRIOR]:
    for reason in [ONE_REASON, THREE_REASONS, SEVEN_REASONS]:
        for words in [LESS_WORDS, MORE_WORDS]:
            combinations.append((prior, reason, words))

# combinations = [
#     (NO_PRIOR, ONE_REASON, LESS_WORDS),     #   Control
#     (NO_PRIOR, ONE_REASON, MORE_WORDS),     #   Testing for more words
#     (NO_PRIOR, THREE_REASONS, LESS_WORDS),  #   Testing for reason count
#     (NO_PRIOR, SEVEN_REASONS, LESS_WORDS),  #   Testing for reason count
#     (LEFT_PRIOR, ONE_REASON, LESS_WORDS),   #   Testing for left prior
#     (RIGHT_PRIOR, ONE_REASON, LESS_WORDS)   #   Testing for right prior
# ]

for combination in combinations:
    prompt = {
        "engine": "text-davinci-003",
        "prompt": "Follow the instructions in order. " + str(CLASSIFICATION + combination[0] + combination[1] + combination[2]),
        "max_tokens": 300,
        "n": 1,
    }
    prompts.append(prompt)