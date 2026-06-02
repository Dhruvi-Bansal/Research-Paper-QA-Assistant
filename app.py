import streamlit as st

from src.ingest import (
    load_documents,
    split_documents
)

from src.retriever import (
    create_vectorstore
)

from src.qa_chain import (
    build_qa_chain
)

from src.utils import (
    save_uploaded_files
)

from src.citations import (
    format_citations
)


# ==================================================
# Page Config
# ==================================================

st.set_page_config(
    page_title="Research Paper QA Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Research Paper QA Assistant")

st.markdown(
    """
    Upload one or more research papers and ask questions.
    
    Features:
    - Multi-PDF Retrieval
    - Conversational Memory
    - Source Citations with Page Numbers
    - Groq + LangChain RAG Pipeline
    """
)

# ==================================================
# Session State Initialization
# ==================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# ==================================================
# Sidebar
# ==================================================

with st.sidebar:

    st.header("Document Management")

    uploaded_files = st.file_uploader(
        "Upload Research Papers",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        if st.button("Process Papers"):

            with st.spinner("Processing PDFs..."):

                pdf_paths = save_uploaded_files(
                    uploaded_files
                )

                documents = load_documents(
                    pdf_paths
                )

                chunks = split_documents(
                    documents
                )

                create_vectorstore(
                    chunks
                )

                # Build QA Chain ONCE
                st.session_state.qa_chain = build_qa_chain()

                st.session_state.vectorstore_ready = True

                # Clear old chat when new papers are uploaded
                st.session_state.messages = []

            st.success(
                f"{len(uploaded_files)} paper(s) processed successfully."
            )

    st.divider()

    if st.button("Clear Chat"):

        st.session_state.messages = []

        st.rerun()

# ==================================================
# Display Chat History
# ==================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )

# ==================================================
# Chat Section
# ==================================================

if st.session_state.vectorstore_ready:

    user_question = st.chat_input(
        "Ask a question about the uploaded papers..."
    )

    if user_question:

        # ------------------------------
        # User Message
        # ------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_question
            }
        )

        with st.chat_message("user"):

            st.markdown(
                user_question
            )

        # ------------------------------
        # Build Chat History String
        # ------------------------------

        history_text = "\n".join(
            [
                f"{msg['role']}: {msg['content']}"
                for msg in st.session_state.messages[:-1]
            ]
        )

        # ------------------------------
        # Assistant Response
        # ------------------------------

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching research papers..."
            ):

                response = (
                    st.session_state.qa_chain.invoke(
                        {
                            "input": user_question,
                            "chat_history": history_text
                        }
                    )
                )

                answer = response["answer"]

                st.markdown(answer)

                # --------------------------
                # Source Citations
                # --------------------------

                citations = format_citations(
                    response["context"]
                )

                if citations:

                    st.markdown(
                        "### 📖 Sources"
                    )

                    for citation in citations:

                        st.markdown(
                            f"- {citation}"
                        )

        # ------------------------------
        # Save Assistant Response
        # ------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

else:

    st.info(
        "Upload and process at least one PDF to begin chatting."
    )