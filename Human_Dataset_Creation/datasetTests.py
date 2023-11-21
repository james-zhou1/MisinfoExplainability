def testExplanationFormatValid(explanation):
    if '. ' in explanation: return True
    print("No period to split the query, trying again...")
    return False

def testClassificationValid(classification):
    valid_classifications = ["true", "mostly-true", "half-true", "mostly-false", "false", "pants-fire"]
    if classification.endswith('.'): classification = classification[:-1]
    if classification in valid_classifications: return True
    print(classification)
    print("Not a valid classification, trying again...")
    return False

def testClassificationCorrect(classification, label):
    if not testClassificationValid(classification):
        return testClassificationValid(classification)
    if label == classification:
        return True
    print("Actual:", label.lower())
    print("Given:", classification)
    print("Not a correct classification, trying again...")
    return False