from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai 
from dotenv import load_dotenv
import os
import shutil

import nltk 
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
from nltk import word_tokenize, sent_tokenize

load_dotenv()

openai.api_key = os.environ['OPENAI_API_KEY']

CHROMA_PATH = "chroma"
DATA_PATH = "data/"


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    print(f"Loaded {len(documents)} documents from {DATA_PATH}.")
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    try:
        if os.path.exists(CHROMA_PATH): #clear db
            shutil.rmtree(CHROMA_PATH)
        
        db = Chroma.from_documents( #create db
            chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
        )
        db.persist()
        print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")
    except Exception as e:
        print(f"Error saving to Chroma: {e}")


if __name__ == "__main__":
    main()