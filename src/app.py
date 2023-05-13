import time
from typing import Callable, Optional
import requests
import numpy as np
from sklearn.neighbors import NearestNeighbors
import json
import logging
import os

EMBEDDING_API_URL = 'http://localhost:3369' #'http://embedtext.com'
# Maximum number of sentences that can be processed at once. Should match the value in the embedding API.
MAX_SENTENCES = 100

# Configure logging
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)

def log_header(text: str):
    logger.info(f'-------- {text} --------')

def load_file(filepath: str):
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the absolute path of the file
    abs_file_path = os.path.join(script_dir, filepath)
    try:
        with open(abs_file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        logger.error(f"The file at {abs_file_path} could not be found.")
        return None
    except json.JSONDecodeError:
        logger.error(f"Could not parse the JSON file at {abs_file_path}.")
        return None

def load_object_data(filename: str):
    return load_file(f'objects/{filename}.json')
    
def load_search_data(filename: str):
    return load_file(f'searches/{filename}.json')
    
def load_instructions(filename: str):
    return load_file(f'instructions/{filename}.json')

# Uses the embedding API to get the embeddings of the instruction and the sentences in the list
# Returns None if there is an error
def get_embeddings(instruction, sentences):
    embeddings = []
    for i in range(0, len(sentences), MAX_SENTENCES):
        subset_sentences = sentences[i:i+MAX_SENTENCES]
        data = json.dumps({'instruction': instruction, 'sentences': subset_sentences})
        headers = {'Content-Type': 'application/json'}
        logger.info(f'Getting embeddings: {instruction}')
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

def test_embeddings(test_strings: list, instruction: str, search_strings: list, consider_negative: bool = False, parse_fn: Optional[Callable] = None):
    """
    Tests a single instruction by getting the embeddings of the test strings and search strings, finding the 
    nearest neighbors for each search string, and comparing the results with the expected results.

    Parameters:
    test_strings (list): List of strings to embed and use as the dataset for the nearest neighbors algorithm.
    instruction (str): The instruction to use for the embeddings.
    search_strings (list): List of tuples, where each tuple has a search string and the expected result.
    consider_negative (bool, optional): Whether to consider negative results (i.e. results trying to game the system) when calculating the score. Defaults to True.
    parse_fn (Callable, optional): Function to parse the search results. Useful if you're representing multiple fields in one string

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
        if parse_fn:
            # If a parse function is provided, use it to parse the results
            nearest_neighbors = [parse_fn(n) for n in nearest_neighbors]
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

        # If consider_negative is True and the "negative" key exists in the dictionary, calculate the negative score
        if consider_negative and "negative" in search_strings[i]:
            for neg in search_strings[i]["negative"]:
                if neg in nearest_neighbors:
                    neg_position = nearest_neighbors.index(neg)
                    neg_score = max(0.1*(len(search_strings[i][1])-1) - 0.1*neg_position, 0)
                    search_scores.append(neg_score)
                    logger.debug(f"Negative score for '{neg}' in search '{search_strings[i][0]}': {neg_score}")

        # Average the scores for the expected results of this search string
        avg_score = sum(search_scores) / len(search_strings[i][1])
        scores.append(avg_score)

    return scores

def test_multiple_instructions(instructions: list, test_strings: list, search_data: list, consider_negative: bool = False, parse_fn: Optional[Callable] = None):
    """
    This function tests multiple instructions by passing each one to test_embeddings and then sorts the instructions 
    based on the average score they received.

    Parameters:
    instructions (list): List of instructions to test.
    test_strings (list): List of strings to embed and use as the dataset for the nearest neighbors algorithm.
    search_data (list): List of tuples, where each tuple has a search string and the expected result.
    consider_negative (bool, optional): Whether to consider negative results (i.e. results trying to game the system) when calculating the score. Defaults to True.
    parse_fn (Callable, optional): Function to parse the search results. Useful if you're representing multiple fields in one string
    """
    instruction_scores = []

    for instruction in instructions:
        logger.info(f'Testing instruction: "{instruction}"')
        scores = test_embeddings(test_strings, instruction, search_data, consider_negative, parse_fn)
        if scores is not None:  # If there was no error in test_embeddings
            avg_score = sum(scores) / len(scores)
            instruction_scores.append((instruction, avg_score))

            # Log the average score for each search string
            for i in range(len(search_data)):
                logger.debug(f"Average score for search string '{search_data[i][0]}': {scores[i]}")

            # Log the overall average score for this instruction
            logger.debug(f"Average score for instruction '{instruction}': {avg_score}")

    # Sort the instructions by their average score, in descending order
    instruction_scores.sort(key=lambda x: x[1], reverse=True)
    logger.debug(f"Instructions from best to worst: {instruction_scores}")
    logger.info(f"Best instruction: {instruction_scores[0]}")


def test_tags():
    log_header("Start test: Tags")
    tags = load_object_data("tags")
    result_strings = [tag['name'] for tag in tags]
    searches = load_search_data("tags")
    search_data = [(search['search'], search['expected']) for search in searches]
    instructions = load_instructions("classify")
    test_multiple_instructions(instructions, result_strings, search_data)
    log_header("End test: Tags")

def test_tags_with_descriptions():
    log_header("Start test: Tags with descriptions")
    tags = load_object_data("tags")
    test_strings = [json.dumps(tag) for tag in tags]
    searches = load_search_data("tags")
    search_data = [(search['search'], search['expected']) for search in searches]
    instructions = load_instructions("classify")
    def parse_result(result):
        # Parse the JSON string and return the name field
        return json.loads(result)["name"]
    test_multiple_instructions(instructions, test_strings, search_data, True, parse_fn=parse_result)
    log_header("End test: Tags with descriptions")

def test_organizations():
    log_header("Start test: Organizations")
    organizations = load_object_data("organizations")
    test_strings = [json.dumps(org) for org in organizations]
    searches = load_search_data("organizations")
    search_data = [(search['search'], search['expected']) for search in searches]
    instructions = load_instructions("classifyWithDeboost")
    def parse_result(result):
        # Parse the JSON string and return the name field
        return json.loads(result)["name"]
    test_multiple_instructions(instructions, test_strings, search_data, True, parse_fn=parse_result)
    log_header("End test: Organizations")

def main():
    log_header("Starting Tests")
    test_tags()
    test_tags_with_descriptions()
    test_organizations()
    log_header("Finished Tests")

try:
    main()
except Exception as e:
    logger.error(f"Error running main: {e}")
