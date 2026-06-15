from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate

from langchain_classic.chains import (
    create_retrieval_chain
)

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)

from src.retriever import get_retriever

load_dotenv()


QA_PROMPT = """
You are a research paper assistant.

Rules:

1. Answer ONLY using retrieved context.
2. Do NOT use external knowledge.
3. If information is missing, say:

'The uploaded papers do not contain enough information to answer this question.'

Context:
{context}

Question:
{input}

Answer:
"""

def build_qa_chain():

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    retriever = get_retriever()

    prompt = ChatPromptTemplate.from_template(
        QA_PROMPT
    )

    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    return retrieval_chain