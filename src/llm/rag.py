"""
Before running this program, you should first have the Milvus and whoosh databases created.
"""
import warnings
warnings.filterwarnings(action="ignore", message="Protobuf gencode version")

import argparse
import time
import json
from typing import List
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from whoosh.index import open_dir
from whoosh.qparser import QueryParser
from whoosh.query import *
from langchain_openai import ChatOpenAI
from openai import APIConnectionError
from langchain.tools import tool
from langchain.agents import create_agent
import numpy as np
from tqdm import tqdm

try:
    from utils.inference_auth_token import get_access_token
except ImportError:
    from src.inference_auth_token import get_access_token

K = 10 # number of top results to return


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a RAG (Retrieval-Augmented Generation) pipeline with a specified strategy and models."
    )

    parser.add_argument(
        "--generation-model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-70B-Instruct",
        help="Name or path of the model used for text generation."
    )

    parser.add_argument(
        "--database-path",
        type=str,
        help="Path to the vector or keyword database."
    )

    parser.add_argument(
        "--question-path",
        type=str,
        default="data/questions.jsonl",
        help="The path to the questions that will be sent to the agent."
    )

    subparsers = parser.add_subparsers(dest='strategy')
    subparsers.required = True
    parser_vector = subparsers.add_parser('vector')
    # parser_vector.add_argument(
    #     "--embedding-model",
    #     type=str,
    #     required=True,
    #     help="Name or path of the model to use for vector generation."
    # )

    subparsers.add_parser('kw')

    return parser.parse_args()

@tool
def get_vector_result(query: str) -> List[str]:
    """
    Get the top k matches of document entries based on the provided keywords.
    Only the query is needed from model output.
    """
    client = QdrantClient(url="http://localhost:13031")
    embedding_model = SentenceTransformer('sentence-transformers/paraphrase-albert-small-v2')

    query_vector = embedding_model.encode([query])[0].tolist()
    
    res = client.query_points(collection_name="documents", 
                              query=query_vector, limit=K, 
                              with_payload=True).points
    res = [point.payload["text"] for point in res]
    print(len(res))
    print(res)
    
    return res

@tool
def get_whoosh_result(keywords: str) -> List[str]:
    """
    Get the top K matches of document entries based on the provided keywords
    (a single string with each keyword separated by spaces).
    Only the keywords are needed from model output.
    """
    ix = open_dir(database_path)

    parser = QueryParser("text", schema=ix.schema)
    query = parser.parse(keywords)

    with ix.searcher() as searcher:
        res = searcher.search(query, limit=K)
        return [entry["text"] for entry in res] # extract the text field


if __name__ == "__main__":
    args = parse_args()
    print(f"Strategy: {args.strategy}")
    print(f"Generation model: {args.generation_model}")
    print(f"Database path: {args.database_path}")
    print(f"Question path: {args.question_path}")

    if args.database_path:
        database_path = args.database_path
    else: # use default paths
        database_path = "kw" if args.strategy == "kw" else "vector.db"

    # Get your access token
    access_token = get_access_token()

    client = ChatOpenAI(
        api_key=access_token,
        base_url="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1",
        model=args.generation_model,
    )

    system_prompt = "Evaluate the following question and try to come up with an one-sentence answer." +\
                    " Only keep the relevant information from the retrieved documents." +\
                    " Write your final answer after '#### '.\n"

    if args.strategy == "vector":
        system_prompt += "\nYou may ask questions by providing a sentence/keywords you want to search for."
    elif args.strategy == "kw":
        system_prompt += "\nYou may ask questions by providing some keywords you want to search for. Note that the " +\
                "search will match ALL the keywords you have provided, so try to give fewer keywords."
            
    agent = create_agent(
        model=client,
        tools=[get_vector_result] if args.strategy == "vector" else [get_whoosh_result],
        prompt=system_prompt,
    )

    times = [] # record time taken
    tokens = [] # number of tokens spent
    model_answers = [] # model's answers to each question
    correct_answers = [] # correct answers to each question
    
    with open(args.question_path, encoding="utf-8") as f:
        contents = f.readlines()
    for i in tqdm(range(len(contents))):
        line = contents[i].strip()

        text = json.loads(line)
        question = text['question']
        keywords = text['keywords']
        origin = text['text']
        correct_answer = text['answer']
        correct_answers.append(correct_answer)

        retry = 0

        prompt = "\nQuestion: " + question + "\n"

        now = time.time()
        while True:
            try:
                retry += 1
                if retry > 5:
                    print(f"Failed to get a valid answer after {retry} retries, moving on...")
                    model_answers.append("Invalid answer")
                    break
                
                response = agent.invoke({"messages": [{"role":"user","content":prompt}]})['messages'][-1]
                delta_time = time.time() - now

                answer = response.content.split("#### ")[-1].strip()
                # Identify if the model is not answering the question but trying to call a tool
                if any(x in answer.lower() for x in ["function", "call", "tool", "search", "query"]):
                    continue
                times.append(delta_time)
                tokens.append(response.usage_metadata["total_tokens"])

                model_answers.append(answer)
                break
            except APIConnectionError as e:
                # network problems, DNS, etc.
                print('Connection error')
                exit(1)
            except KeyError as e:
                print(f"Error getting structured output, retrying...")
            except Exception as e:
                print(f"Error: {e}, retrying...")

    for i in range(len(model_answers)):
        print(f"Model answer: {model_answers[i]};\ncorrect answer: {correct_answers[i]}")
    print(f"Average time: {np.mean(times)}" +\
      f" +- {np.std(times)}, average number of tokens: {np.mean(tokens)} +- {np.std(tokens)}")