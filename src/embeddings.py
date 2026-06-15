from streamlit import cache_resource
from langchain_community.embeddings import HuggingFaceEmbeddings


@cache_resource
def get_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )