import requests
import numpy as np
from sklearn.neighbors import NearestNeighbors
import json
import logging

EMBEDDING_API_URL = 'http://localhost:3369' #'http://embedtext.com'

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)

# Uses the embedding API to get the embeddings of the instruction and the sentences in the list
# Returns None if there is an error
def get_embeddings(instruction, sentences):
    data = json.dumps({'instruction': instruction, 'sentences': sentences})
    headers = {'Content-Type': 'application/json'}
    logger.info(f'Getting embeddings with this data: {data}')
    try:
        response = requests.post(EMBEDDING_API_URL, data=data, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting embeddings: {e}")
        return None
    try:
        return np.array(response.json()['embeddings'])
    except Exception as e:
        logger.error(f"Error converting response to numpy array: {e}")
        return None

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

def main():
    logger.info('Starting tests...')
     # Test 1: Tags
    instruction1 = "Represent the text for classification"
    test_strings1 = [
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
        "Governacne",
        "Project Management",
        "Productivity",
        "Documentation"
    ]
    search_strings1 = ["ai", "computer science", "business", "chicken", "entrepreneurship", "entrepreneur", "entre"]
    test_embeddings(test_strings1, instruction1, search_strings1)

    # Test 2: Plain strings
    instruction2 = "Represent the text for classification"
    test_strings2 = [
        "The quick brown fox jumps over the lazy dog",
        "The fast brown fox jumps over the lazy dog",
        # ...
        "C",
    ]
    search_strings2 = ["fox", "pen", "dog", "the", "a"]
    test_embeddings(test_strings2, instruction2, search_strings2)

    # Test 3: JSON strings
    instruction3 = "Represent the title and description for classification"
    test_strings3 = [
        json.dumps({
            "title": "All about penguins",
            "description": "Penguins are birds. They are black and white. When it is cold, they huddle together.",
        }),
        # ...
    ]
    search_strings3 = ["penguins", "bake cake", "dinosaurs", "birds", "baking"]
    test_embeddings(test_strings3, instruction3, search_strings3)

    # Test 4: Same string multiple times
    instruction4 = "Represent the text for classification"
    test_string4 = "The quick brown fox jumps over the lazy dog"
    embeddings4 = np.array([get_embeddings(instruction4, test_string4) for _ in range(3)])
    similarities4 = [np.dot(embeddings4[0], embeddings4[i]) for i in range(3)]
    logger.info(f'Similarities for test 4: {similarities4}')
    logger.info('Tests complete!')

try:
    main()
except Exception as e:
    logger.error(f"Error running main: {e}")
