from langchain_community.vectorstores import FAISS

from src.embeddings import (
    get_embedding_model
)

VECTOR_DB_PATH = "vectorstore"


def create_vectorstore(chunks):
    embeddings = get_embedding_model()

    db = FAISS.from_documents(
        chunks,
        embeddings
    )

    db.save_local(VECTOR_DB_PATH)

    return db


def load_vectorstore():
    embeddings = get_embedding_model()

    db = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db


def get_retriever():
    db = load_vectorstore()

    return db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "score_threshold":0.5,
        "fetch_k": 10
    }
    )