from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(pdf_paths):

    documents = []

    for pdf_path in pdf_paths:

        loader = PyPDFLoader(pdf_path)

        docs = loader.load()

        for doc in docs:

            doc.metadata["paper"] = pdf_path.split("/")[-1]

        documents.extend(docs)

    return documents


def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_documents(documents)