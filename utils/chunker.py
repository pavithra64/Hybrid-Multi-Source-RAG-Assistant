from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import *


def split_text(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    return splitter.split_text(text)