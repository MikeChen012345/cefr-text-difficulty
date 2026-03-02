# This file contains code for ingesting text data into a vector database (Qdrant) for retrieval-augmented generation (RAG) experiments.
import warnings

warnings.filterwarnings(action="ignore", message="Protobuf gencode version")

from typing import List
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import sys
import os

if os.getcwd().endswith("llm"):
    sys.path.append("..")  # Add parent directory to path for imports
elif os.getcwd().endswith("src"):
    sys.path.append(".")  # Ensure current directory is in path for imports
elif os.getcwd().endswith("cefr-text-difficulty"):
    # else if running from project root
    sys.path.append("./src")  # Add src directory to path for imports
else:
    raise ValueError("Unexpected working directory; cannot locate config.yaml")

from data_loader import load_data_split

def ingest_vector(df: pd.DataFrame, dataset_key: str) -> None:
    embedding_model = SentenceTransformer('sentence-transformers/paraphrase-albert-small-v2')
    vector_size = embedding_model.get_sentence_embedding_dimension()

    client = QdrantClient(url="http://localhost:13031")

    vector_params = VectorParams(
        size=vector_size,
        distance=Distance.COSINE
    )

    client.recreate_collection(
        collection_name=dataset_key,
        vectors_config=vector_params,
    )

    if "text" not in df.columns:
        raise ValueError(f"DataFrame must contain a 'text' column")

    batch_size = 128
    print(f"Ingesting in batches of {batch_size}")

    for b in tqdm(range((len(df) - 1) // batch_size + 1)):
        start = b * batch_size
        end = min((b + 1) * batch_size, len(df))
        batch_df = df.iloc[start:end]
        batch_docs = batch_df["text"].tolist()
        vectors = embedding_model.encode_document(batch_docs)
        # vectors is a list of tensors/arrays; coerce each item to plain list[float]
        points = []
        for j, vec in enumerate(vectors):
            flat_vec = [float(x) for x in vec]
            points.append(PointStruct(
                id=start + j,
                vector=flat_vec,
                payload={"text": batch_docs[j], "cefr_level": batch_df.iloc[j]["cefr_level"]}
            ))

        operation_info = client.upsert(
            collection_name=dataset_key,
            wait=True,
            points=points
        )


def main():
    for dataset_key in ["cefr_sp_en", "readme_en"]:
        print(f"Ingesting dataset_key={dataset_key}")
        train_df, test_df = load_data_split(dataset_key=dataset_key)
        # Only ingest the training set to avoid data leakage from test set.
        ingest_vector(train_df, dataset_key=dataset_key)


if __name__ == "__main__":
    main()