import json
import openai

class OpenAIModel:
    def __init__(self, api_key, engine="gpt-3.5-turbo", system_prompt="", name=""):
        self.api_key = api_key
        self.engine = engine
        self.system_prompt = system_prompt
        self.name = name
        openai.api_key = self.api_key

    # def get_label(self, statement):
    #     full_prompt = f"{self.system_prompt} {statement}."
    #     label = openai.Completion.create(
    #         engine=self.engine,
    #         prompt=full_prompt,
    #         max_tokens=60
    #     ).choices[0].text.strip()
    #     return label

    def get_response(self, text, max_tokens=1000, n=1):
        full_prompt = f"{self.system_prompt} {text}."
        response = openai.ChatCompletion.create(
            model=self.engine,
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=max_tokens,
            n=n,
        )
        cost = response['usage']['total_tokens']
        text_response = response.choices[0]['message']['content'].strip()
        return self.name, full_prompt, text_response, cost
    

with open('../key.env', 'r') as file:
    API_KEY = file.read().strip().split('=')[1]
openai.api_key = API_KEY

labeler = OpenAIModel(API_KEY, system_prompt="Classify the following statement as true, mostly-true, half-true, mostly-false, false, or pants-fire. Do not say anything else. Do not restate the prompt. ", name="labeler")

#   Create all explainer models
CLASSIFICATION = "First, classify the following statement in one word as true, mostly-true, half-true, mostly-false, false, or pants-fire, without additional words. "

NO_PRIOR = ""
LEFT_PRIOR = "Second, introduce the text as being from CNN. "
RIGHT_PRIOR = "Second, introduce the text as being from Fox News. "

# ONE_REASON = "Third, give one reason explaining your classification. "
# THREE_REASONS = "Third, give three reasons explaining your classification. "
# SEVEN_REASONS = "Third, give seven reasons explaining your classification. "

#   We are going to test if directly prompting  the model to scale their confidence will make a difference.
ONE_REASON = "Third, give one reason explaining your classification. Scale the confidence of your explanation to how certain you are of the classification. "
THREE_REASONS = "Third, give three reasons explaining your classification. Scale the confidence of your explanation to how certain you are of the classification. "
SEVEN_REASONS = "Third, give seven reasons explaining your classification. Scale the confidence of your explanation to how certain you are of the classification. "

ONE_REASON = "Third, give one reason explaining your classification. " + "Give your response less confidently if you are not confident in the classification. Give your response more confidently if you are more confident in the classification."
THREE_REASONS = "Third, give three reasons explaining your classification. " + "Give your response less confidently if you are not confident in the classification. Give your response more confidently if you are more confident in the classification."
SEVEN_REASONS = "Third, give seven reasons explaining your classification. " + "Give your response less confidently if you are not confident in the classification. Give your response more confidently if you are more confident in the classification."

ONE_REASON = "Third, give one reason explaining your classification. " + "Give your response less confidently if you are not confident in the classification. Give your response more confidently if you are more confident in the classification." + "For example, a less confident responses may include tentative phrases like I think or My best guess"
THREE_REASONS = "Third, give three reasons explaining your classification. " + "Give your response less confidently if you are not confident in the classification. Give your response more confidently if you are more confident in the classification." + "For example, a less confident responses may include tentative phrases like I think or My best guess"
SEVEN_REASONS = "Third, give seven reasons explaining your classification. " + "Give your response less confidently if you are not confident in the classification. Give your response more confidently if you are more confident in the classification." + "For example, a less confident responses may include tentative phrases like I think or My best guess"

description = "An explanation is considered confident when it is delivered with a strong sense of certainty and authority. This involves presenting information in a clear, direct manner, often supported by concrete evidence or well-established knowledge. A confident explanation doesn't hedge with phrases like 'might be' or 'could be,' but instead uses definitive language. The speaker or writer demonstrates a deep understanding of the subject, allowing them to articulate points with precision and conviction. This kind of explanation typically leaves little room for doubt, instilling trust in the audience that the information is accurate and well-founded. Importantly, confidence in an explanation does not imply arrogance or inflexibility; it can coexist with openness to new ideas and perspectives, but it fundamentally conveys a strong belief in the validity of the information presented."

ONE_REASON = "Third, give one reason explaining your classification. " + "Give your response less confidently if you are not confident in the classification. Give your response more confidently if you are more confident in the classification." + "For example, a less confident responses may include tentative phrases like I think or My best guess" + description
THREE_REASONS = "Third, give three reasons explaining your classification. " + "Give your response less confidently if you are not confident in the classification. Give your response more confidently if you are more confident in the classification." + "For example, a less confident responses may include tentative phrases like I think or My best guess" + description
SEVEN_REASONS = "Third, give seven reasons explaining your classification. " + "Give your response less confidently if you are not confident in the classification. Give your response more confidently if you are more confident in the classification." + "For example, a less confident responses may include tentative phrases like I think or My best guess" + description


LESS_WORDS = "Fourth, make sure that each reason is concise at around 50 words. "
MORE_WORDS = "Fourth, make sure that each reason is verbose at around 150 words. "

controlExplainer = OpenAIModel(API_KEY, system_prompt=CLASSIFICATION+NO_PRIOR+ONE_REASON+LESS_WORDS, name="control")
leftExplainer = OpenAIModel(API_KEY, system_prompt=CLASSIFICATION+LEFT_PRIOR+ONE_REASON+LESS_WORDS, name="left")
rightExplainer = OpenAIModel(API_KEY, system_prompt=CLASSIFICATION+RIGHT_PRIOR+ONE_REASON+LESS_WORDS, name="right")
threeExplainer = OpenAIModel(API_KEY, system_prompt=CLASSIFICATION+NO_PRIOR+THREE_REASONS+LESS_WORDS, name="three")
sevenExplainer = OpenAIModel(API_KEY, system_prompt=CLASSIFICATION+NO_PRIOR+SEVEN_REASONS+LESS_WORDS, name="seven")
moreExplainer = OpenAIModel(API_KEY, system_prompt=CLASSIFICATION+NO_PRIOR+ONE_REASON+MORE_WORDS, name="more")
explainers = [controlExplainer]#, leftExplainer, rightExplainer, threeExplainer, sevenExplainer, moreExplainer]

#   Create all evaluator models
# CLASSIFICATION = "Consider the following excerpt, followed by the explanation of the excerpt. If there are multiple reasons given, treat them all as one large explanation. On a scale from 0-100, rate how persuasive this explanation is. Don't include any other text after the number. This means no explanations for your responses. "

# CLASSIFICATION = "Consider the following excerpt, followed by the explanation of the excerpt. If there are multiple reasons given, treat them all as one large explanation. On a scale from 0-100, rate how confident this explanation is. Don't include any other text after the number. This means no explanations for your responses. "

CLASSIFICATION = "Consider the following excerpt, followed by the explanation of the excerpt. If there are multiple reasons given, treat them all as one large explanation. On a scale from 0-100, rate how persuasive this explanation is. Don't include any other text after the number. This means no explanations for your responses. "


LEFT_BIAS = "Give your evaluation from the perspective of someone who is left-leaning. "
RIGHT_BIAS = "Give your evaluation from the perspective of someone who is right-leaning. "

controlEvaluator = OpenAIModel(API_KEY, system_prompt=CLASSIFICATION, name="control")
leftEvaluator = OpenAIModel(API_KEY, system_prompt=CLASSIFICATION+LEFT_BIAS, name="left")
rightEvaluator = OpenAIModel(API_KEY, system_prompt=CLASSIFICATION+RIGHT_BIAS, name="right")
evaluators = [controlEvaluator]#, leftEvaluator, rightEvaluator]