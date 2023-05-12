import requests
import numpy as np
from sklearn.neighbors import NearestNeighbors
import json
import logging

EMBEDDING_API_URL = 'http://localhost:3369' #'http://embedtext.com'
# Maximum number of sentences that can be processed at once. Should match the value in the embedding API.
MAX_SENTENCES = 100

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
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

# Gets embeddings, finds the k nearest neighbors, and prints them.
# Used to simulate what a search result would return.
def test_embeddings(test_strings, instruction, search_strings, k=5):
    embeddings = get_embeddings(instruction, test_strings)
    search_embeddings = get_embeddings(instruction, search_strings)

    # Remove any None values that might have been returned by get_embeddings
    embeddings = [e for e in embeddings if e is not None]
    search_embeddings = [e for e in search_embeddings if e is not None]

    try:
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(np.array(embeddings))
        distances, indices = nbrs.kneighbors(np.array(search_embeddings))
    except Exception as e:
        logger.error(f"Error fitting NearestNeighbors or getting neighbors: {e}")
        return

    for i in range(len(search_strings)):
        nearest_neighbors = [test_strings[index] for index in indices[i]]
        logger.info(f'Nearest neighbors for "{search_strings[i]}": {nearest_neighbors}')

def test_1():
    logger.info('-------- Starting Test 1 --------')
    instruction = "Represent the text for classification"
    test_strings = [
        "Low-Code",
        "Product Development",
        "Idea Generation",
        "Security",
        "Software Development",
        "Open Source",
        "Blockchain",
        "Oracle",
        "Compliance",
        "Analytics",
        "Web3",
        "Decentralized Finance (DeFi)",
        "Governance",
        "Project Management",
        "Productivity",
        "Documentation"
    ]
    search_strings = ["ai", "computer science", "business", "chicken", "entrepreneurship", "entrepreneur", "entre"]
    test_embeddings(test_strings, instruction, search_strings)
    logger.info('-------- Test 1 Complete --------')

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
    test_1()
    test_2()
    test_3()
    test_4()
    logger.info('-------- Tests Complete --------')

try:
    main()
except Exception as e:
    logger.error(f"Error running main: {e}")
