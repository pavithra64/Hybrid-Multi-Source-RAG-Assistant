import pickle
import faiss

from sentence_transformers import SentenceTransformer

from config import *

index = faiss.read_index(
    "vectorstore/faiss.index"
)

with open(
    "vectorstore/metadata.pkl",
    "rb"
) as f:

    metadata = pickle.load(f)

model = SentenceTransformer(
    EMBEDDING_MODEL
)


def retrieve(query):

    query_embedding = model.encode(
        [query]
    )

    distances, indices = index.search(
        query_embedding,
        TOP_K
    )

    docs = []

    for idx in indices[0]:

        docs.append(
            metadata[idx]
        )

    return docs