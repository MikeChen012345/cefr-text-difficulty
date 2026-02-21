import argparse
import json
import os
import warnings

warnings.filterwarnings(action="ignore", message="Protobuf gencode version")

from typing import List
from langchain_text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from whoosh.index import create_in
from whoosh.fields import *

def ingest_vector(documents: list[str], db_path: str, vector_size: int) -> None:
    client = QdrantClient(url="http://localhost:13031")

    vector_params = VectorParams(
        size=vector_size,
        distance=Distance.COSINE
    )

    client.recreate_collection(
        collection_name="documents",
        vectors_config=vector_params,
    )

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50,
                                              separators=["\n\n", "\n", "."])

    embedding_model = SentenceTransformer('sentence-transformers/paraphrase-albert-small-v2')

    docs: List[str] = []
    for doc in documents:
        with open(doc, encoding="utf-8") as f:
            content = f.readlines()
        for i in tqdm(range(len(content))):
            text = json.loads(content[i])["text"]
            chunks = splitter.split_text(text)
            for chunk in chunks:
                docs.append(chunk)
    
    print(f"Number of chunks: {len(docs)}")

    batch_size = 128
    print(f"Ingesting in batches of {batch_size}")

    for b in tqdm(range((len(docs) - 1) // batch_size + 1)):
        start = b * batch_size
        end = min((b + 1) * batch_size, len(docs))
        batch_docs = docs[start:end]
        vectors = embedding_model.encode_document(batch_docs)
        # vectors is a list of numpy arrays or lists; ensure each item is a plain list of floats
        points = []
        for j, vec in enumerate(vectors):
            # If vec is a numpy array, convert to list; if it's already a list, create a shallow copy
            try:
                flat_vec = list(vec)
            except TypeError:
                # Fallback: attempt to coerce by iterating
                flat_vec = [float(x) for x in vec]
            points.append(PointStruct(
                id=start + j,
                vector=flat_vec,
                payload={"text": batch_docs[j]}
            ))

        operation_info = client.upsert(
            collection_name="documents",
            wait=True,
            points=points
        )


def ingest_text(documents: List[str], db_path: str) -> None:
    # Store the text field so stored_fields() and searcher.documents() return the content
    schema = Schema(path=ID, text=TEXT(stored=True))
    if not os.path.exists(db_path):
        os.mkdir(db_path)
    ix = create_in(db_path, schema)
    writer = ix.writer()

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50,
                                              separators=["\n\n", "\n", "."])
    
    docs: List[str] = []
    for doc in documents:
        with open(doc, encoding="utf-8") as f:
            for line in f:
                text = json.loads(line)["text"]
                chunks = splitter.split_text(text)
                for chunk in chunks:
                    if not isinstance(chunk, str): # Ensure chunk is unicode
                        raise ValueError("Chunk is not a string")
                    docs.append(chunk)
    
    for i in tqdm(range(len(docs))):
        chunk = docs[i]
        writer.add_document(path=f"doc_{i}", text=chunk)
    writer.commit()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest vectors into a database using a specified model and dataset."
    )

    subparsers = parser.add_subparsers(dest='strategy')
    subparsers.required = True
    parser_vector = subparsers.add_parser('vector')
    # parser_vector.add_argument(
    #     "--model",
    #     type=str,
    #     required=True,
    #     help="Name or path of the model to use for vector generation."
    # )
    parser_vector.add_argument(
        "--embedding-size", 
        type=int, 
        default=768,
        help="Embedding size of the vectors.")
    
    # Accept dataset/database args after the subcommand (so calls like
    # `ingest.py vector --dataset-path ... --database-path ...` work):
    parser_vector.add_argument(
        "--dataset-path",
        type=str,
        default="data/data1.jsonl",
        help="Path to the input dataset."
    )
    parser_vector.add_argument(
        "--database-path",
        type=str,
        default="vector",
        help="Path to the output database file." 
    )

    parser_kw = subparsers.add_parser('kw')
    parser_kw.add_argument(
        "--dataset-path",
        type=str,
        default="data/data1.jsonl",
        help="Path to the input dataset."
    )
    parser_kw.add_argument(
        "--database-path",
        type=str,
        default="kw",
        help="Path to the output database folder."
    )
    
    return parser.parse_args()

def main():
    args = parse_args()

    # Dispatch to the selected ingest strategy
    if args.strategy == 'vector':
        # parser stored dataset/database on the subparser
        ingest_vector([args.dataset_path], args.database_path, args.embedding_size)
    elif args.strategy == 'kw':
        ingest_text([args.dataset_path], args.database_path)


if __name__ == "__main__":
    main()