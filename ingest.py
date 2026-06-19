import os
import pickle
import faiss
import numpy as np
import sys

# Adds the current directory to the Python search path
sys.path.append(r"C:\Users\palan\Downloads\GenAI-RAG-pipeline")

from sentence_transformers import SentenceTransformer

from utils.loaders import load_pdf
from utils.loaders import load_txt

from utils.chunker import split_text

from config import *

documents = []

pdf_folder = r"C:\Users\palan\Downloads\GenAI-RAG-pipeline\vectorstore\data\pdfs"

for file in os.listdir(pdf_folder):

    if file.endswith(".pdf"):

        text = load_pdf(
            os.path.join(pdf_folder, file)
        )

        chunks = split_text(text)

        for chunk in chunks:

            documents.append({
                "text": chunk,
                "source": file
            })

txt_folder = r"C:\Users\palan\Downloads\GenAI-RAG-pipeline\vectorstore\data\texts"

for file in os.listdir(txt_folder):

    if file.endswith(".txt"):

        text = load_txt(
            os.path.join(txt_folder, file)
        )

        chunks = split_text(text)

        for chunk in chunks:

            documents.append({
                "text": chunk,
                "source": file
            })

texts = [
    doc["text"]
    for doc in documents
]

model = SentenceTransformer(
    EMBEDDING_MODEL
)

embeddings = model.encode(
    texts,
    show_progress_bar=True
)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)

index.add(
    np.array(embeddings)
)

os.makedirs(
    "vectorstore",
    exist_ok=True
)

faiss.write_index(
    index,
    "vectorstore/faiss.index"
)

with open(
    "vectorstore/metadata.pkl",
    "wb"
) as f:

    pickle.dump(
        documents,
        f
    )

print("Index Created")
