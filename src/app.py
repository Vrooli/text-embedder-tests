import requests
import numpy as np
from sklearn.neighbors import NearestNeighbors
import json
import logging

EMBEDDING_API_URL = 'http://localhost:3369' #'http://embedtext.com'
# Maximum number of sentences that can be processed at once. Should match the value in the embedding API.
MAX_SENTENCES = 100

# Configure logging
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)

# Uses the embedding API to get the embeddings of the instruction and the sentences in the list
# Returns None if there is an error
def get_embeddings(instruction, sentences):
    embeddings = []
    for i in range(0, len(sentences), MAX_SENTENCES):
        subset_sentences = sentences[i:i+MAX_SENTENCES]
        data = json.dumps({'instruction': instruction, 'sentences': subset_sentences})
        headers = {'Content-Type': 'application/json'}
        logger.info(f'Getting embeddings with this data: {data}')
        try:
            response = requests.post(EMBEDDING_API_URL, data=data, headers=headers)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting embeddings: {e}")
            return None
        try:
            embeddings.extend(response.json()['embeddings'])
        except Exception as e:
            logger.error(f"Error converting response to numpy array: {e}")
            return None
    return np.array(embeddings)

def test_embeddings(test_strings: list, instruction: str, search_strings: list):
    """
    Tests a single instruction by getting the embeddings of the test strings and search strings, finding the 
    nearest neighbors for each search string, and comparing the results with the expected results.

    Parameters:
    test_strings (list): List of strings to embed and use as the dataset for the nearest neighbors algorithm.
    instruction (str): The instruction to use for the embeddings.
    search_strings (list): List of tuples, where each tuple has a search string and the expected result.

    Returns:
    list: List of scores for each search string. Each score is the average score for the expected results of that search string.
    """
    embeddings = get_embeddings(instruction, test_strings)
    search_embeddings = get_embeddings(instruction, [s[0] for s in search_strings])

    # Remove any None values that might have been returned by get_embeddings
    embeddings = [e for e in embeddings if e is not None]
    search_embeddings = [e for e in search_embeddings if e is not None]

    nbrs = NearestNeighbors(n_neighbors=len(test_strings), algorithm='ball_tree').fit(np.array(embeddings))
    distances, indices = nbrs.kneighbors(np.array(search_embeddings))

    scores = []
    for i in range(len(search_strings)):
        nearest_neighbors = [test_strings[index] for index in indices[i]]
        logger.debug(f'Nearest neighbors for "{search_strings[i][0]}": {nearest_neighbors}')

        # Calculate the score for each expected result
        search_scores = []
        for j in range(len(search_strings[i][1])):
            if search_strings[i][1][j] in nearest_neighbors:
                position = nearest_neighbors.index(search_strings[i][1][j])
                score = max(1 - 0.1*position, 0)
            else:
                score = 0
            search_scores.append(score)
            logger.debug(f"Score for '{search_strings[i][1][j]}' in search '{search_strings[i][0]}': {score}")

        # Average the scores for the expected results of this search string
        avg_score = sum(search_scores) / len(search_strings[i][1])
        scores.append(avg_score)

    return scores

def test_multiple_instructions(instructions: list, test_strings: list, search_strings: list):
    """
    This function tests multiple instructions by passing each one to test_embeddings and then sorts the instructions 
    based on the average score they received.

    Parameters:
    instructions (list): List of instructions to test.
    test_strings (list): List of strings to embed and use as the dataset for the nearest neighbors algorithm.
    search_strings (list): List of tuples, where each tuple has a search string and the expected result.
    """
    instruction_scores = []

    for instruction in instructions:
        logger.info(f'Testing instruction: "{instruction}"')
        scores = test_embeddings(test_strings, instruction, search_strings)
        if scores is not None:  # If there was no error in test_embeddings
            avg_score = sum(scores) / len(scores)
            instruction_scores.append((instruction, avg_score))

            # Log the average score for each search string
            for i in range(len(search_strings)):
                logger.debug(f"Average score for search string '{search_strings[i][0]}': {scores[i]}")

            # Log the overall average score for this instruction
            logger.debug(f"Average score for instruction '{instruction}': {avg_score}")

    # Sort the instructions by their average score, in descending order
    instruction_scores.sort(key=lambda x: x[1], reverse=True)
    logger.debug(f"Instructions from best to worst: {instruction_scores}")
    logger.info(f"Best instruction: {instruction_scores[0]}")


def test_tags():
    # Test 1: Tags
    result_strings = sorted([
        "Analytics",
        "Artificial Intelligence (AI)",
        "Blockchain",
        "Compliance",
        "Cryptocurrency",
        "Cybersecurity",
        "Data Science",
        "Decentralized Finance (DeFi)",
        "Documentation",
        "E-Commerce",
        "Fintech",
        "GPT-3",
        "Governance",
        "Idea Generation",
        "Internet of Things (IoT)",
        "Low-Code",
        "Machine Learning",
        "Marketing",
        "Open Source",
        "Oracle",
        "Product Development",
        "Productivity",
        "Project Management",
        "Robotics",
        "Security",
        "Software Development",
        "Startups",
        "Web3",
        "Web Development",
    ])
    # Search strings with the expected results
    search_strings = [
        ("ai", ["Artificial Intelligence (AI)", "Machine Learning", "GPT-3"]),
        ("computer science", ["Software Development", "Open Source", "Low-Code"]),
        ("business", ["Product Development", "Productivity", "Startups"]),
        ("blockchain", ["Blockchain", "Decentralized Finance (DeFi)", "Cryptocurrency"]),
        ("documentation", ["Documentation", "Project Management", "Product Development"]),
        ("internet", ["Internet of Things (IoT)", "Web Development", "E-Commerce"]),
        ("security", ["Security", "Cybersecurity", "Compliance"]),
        ("finance", ["Fintech", "Cryptocurrency", "Decentralized Finance (DeFi)"]),
        ("data", ["Data Science", "Machine Learning", "Analytics"]),
        ("marketing", ["Marketing", "E-Commerce", "Startups"]),
    ]
    # Both good and bad instructions are included in this test
    instructions = [
        "Represent the text for classification",
        "Embed this text",
        "Generate a semantic representation of this text",
        "Classify by category",
        "Classify by tag",
        "Classify by alphabetical order",
        "Generate a nonsensical representation of this text",
        "Disregard the semantics of this text",
        "Treat this text as a random string of characters",
        "This text is a shopping list",
        "This text is a mathematical equation",
        "This text is a song lyric",
        "This text is a recipe",
        "Randomize this text",
        "Visualize this text as a landscape",
    ]
    test_multiple_instructions(instructions, result_strings, search_strings)

def test_2():
    logger.info('-------- Starting Test 2 --------')
    instruction = "Represent the text for classification"
    test_strings = [
        "The quick brown fox jumps over the lazy dog",
        "The fast brown fox jumps over the lazy dog",
        "C",
    ]
    search_strings = ["fox", "pen", "dog", "the", "a"]
    test_embeddings(test_strings, instruction, search_strings)
    logger.info('-------- Test 2 Complete --------')

def test_3():
    logger.info('-------- Starting Test 3 --------')
    instruction = "Represent the title and description for classification"
    test_strings = [
        json.dumps({
            "title": "All about penguins",
            "description": "Penguins are birds. They are black and white. When it is cold, they huddle together.",
        }),
        # ...
    ]
    search_strings = ["penguins", "bake cake", "dinosaurs", "birds", "baking"]
    test_embeddings(test_strings, instruction, search_strings)
    logger.info('-------- Test 3 Complete --------')

def test_4():
    logger.info('-------- Starting Test 4 --------')
    instruction = "Represent the text for classification"
    test_string = "The quick brown fox jumps over the lazy dog"
    embeddings = np.array([get_embeddings(instruction, test_string) for _ in range(3)])
    similarities = [np.dot(embeddings[0], embeddings[i]) for i in range(3)]
    logger.info(f'Similarities for test 4: {similarities}')
    logger.info('-------- Test 4 Complete --------')

def main():
    logger.info('-------- Starting Tests --------')
    test_tags()
    test_2()
    test_3()
    test_4()
    logger.info('-------- Tests Complete --------')

try:
    main()
except Exception as e:
    logger.error(f"Error running main: {e}")
