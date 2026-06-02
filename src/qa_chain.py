from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)

from src.retriever import get_retriever

load_dotenv()


def build_qa_chain():

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    retriever = get_retriever()

    prompt = ChatPromptTemplate.from_template(
        """
        You are a research assistant.

        Use the retrieved context and chat history.

        Chat History:
        {chat_history}

        Context:
        {context}

        Question:
        {input}

        Answer:
        """
    )

    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    return chain