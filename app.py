import streamlit as st

from src.ingest import (
    load_documents,
    split_documents
)
from src.guardrails import (
    validate_response_scope
)

from src.retriever import (
    create_vectorstore
)
from src.guardrails import (
    is_prompt_injection
)
from src.qa_chain import (
    build_qa_chain
)

from src.summarizer import (
    summarize_single_paper,
    summarize_all_papers,
    compare_papers
)

from src.utils import (
    save_uploaded_files
)

from src.citations import (
    format_citations
)


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Research Paper QA Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Research Paper QA Assistant")

st.markdown(
    """
    Upload one or more research papers and:

    - Ask questions using RAG
    - Generate detailed paper summaries
    - View source citations
    - Chat across multiple PDFs
    """
)

# ==================================================
# SESSION STATE
# ==================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "documents" not in st.session_state:
    st.session_state.documents = None

# Cache for individual paper summaries
# Example:
# {
#   "paper1.pdf": "...summary...",
#   "paper2.pdf": "...summary..."
# }
if "summary_cache" not in st.session_state:
    st.session_state.summary_cache = {}

# Cache for comparative analysis
if "comparison_cache" not in st.session_state:
    st.session_state.comparison_cache = None

# Optional: Cache for all-paper summaries
if "all_summaries_cache" not in st.session_state:
    st.session_state.all_summaries_cache = None

    
# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.header("📄 Document Upload")

    uploaded_files = st.file_uploader(
        "Upload PDF Research Papers",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        if st.button("Process Papers"):

            with st.spinner(
                "Processing papers..."
            ):

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
                

                st.session_state.qa_chain = (
                    build_qa_chain()
                )

                st.session_state.documents = (
                    documents
                )

                st.session_state.vectorstore_ready = True

                # Clear chat
                st.session_state.messages = []

                # Clear cached summaries
                st.session_state.summary_cache = {}
                st.session_state.comparison_cache = None
                st.session_state.all_summaries_cache = None
    st.divider()

    if st.button("Clear Chat"):

        st.session_state.messages = []

        st.rerun()

# ==================================================
# MODE SELECTION
# ==================================================

mode = st.radio(
    "Select Mode",
    [
        "Question Answering",
        "Paper Summary"
    ],
    horizontal=True
)

# ==================================================
# SUMMARY MODE
# ==================================================

if mode == "Paper Summary":

    if not st.session_state.documents:

        st.info(
            "Upload and process PDFs first."
        )

    else:

        summary_mode = st.radio(
            "Summary Type",
            [
                "Single Paper Summary",
                "All Papers Summary",
                "Comparative Summary"
            ]
        )

        paper_names = sorted(
            {
                doc.metadata.get(
                    "paper",
                    "Unknown Paper"
                )
                for doc
                in st.session_state.documents
            }
        )

        # ----------------------------------
        # SINGLE PAPER SUMMARY
        # ----------------------------------

        if summary_mode == "Single Paper Summary":

            selected_paper = st.selectbox(
                "Select Paper",
                paper_names
            )

            if st.button(
                "Generate Summary"
            ):

                if (
                    selected_paper
                    not in st.session_state.summary_cache
                ):

                    with st.spinner(
                        "Generating summary..."
                    ):

                        summary = (
                            summarize_single_paper(
                                st.session_state.documents,
                                selected_paper
                            )
                        )

                        st.session_state.summary_cache[
                            selected_paper
                        ] = summary

                st.markdown(
                    f"# 📄 {selected_paper}"
                )

                st.markdown(
                    st.session_state.summary_cache[
                        selected_paper
                    ]
                )

        # ----------------------------------
        # ALL PAPERS SUMMARY
        # ----------------------------------

        elif summary_mode == "All Papers Summary":

            if st.button(
                "Generate Summaries"
            ):

                if (
                    st.session_state.all_summaries_cache
                    is None
                ):

                    with st.spinner(
                        "Generating summaries..."
                    ):

                        st.session_state.all_summaries_cache = (
                            summarize_all_papers(
                                st.session_state.documents
                            )
                        )

                summaries = (
                    st.session_state.all_summaries_cache
                )

                for paper_name, summary in summaries.items():

                    st.markdown(
                        f"# 📄 {paper_name}"
                    )

                    st.markdown(summary)

                    st.divider()
        # ----------------------------------
        # COMPARATIVE SUMMARY
        # ----------------------------------

        else:

            if len(paper_names) < 2:

                st.warning(
                    "Upload at least 2 papers."
                )

            else:

                if st.button(
                    "Compare Papers"
                ):

                    if (
                        st.session_state.comparison_cache
                        is None
                    ):

                        with st.spinner(
                            "Comparing papers..."
                        ):

                            st.session_state.comparison_cache = (
                                compare_papers(
                                    st.session_state.documents
                                )
                            )

                    st.markdown(
                        "# 📊 Comparative Analysis"
                    )

                    st.markdown(
                        st.session_state.comparison_cache
                    )
# ==================================================
# QUESTION ANSWERING MODE
# ==================================================

if mode == "Question Answering":

    if not st.session_state.vectorstore_ready:

        st.info(
            "Upload and process PDFs to start chatting."
        )

    else:

        # --------------------------------------
        # DISPLAY CHAT HISTORY
        # --------------------------------------

        for message in st.session_state.messages:

            with st.chat_message(
                message["role"]
            ):

                st.markdown(
                    message["content"]
                )

        # --------------------------------------
        # USER INPUT
        # --------------------------------------

        user_question = st.chat_input(
            "Ask a question about the papers..."
        )

        if user_question:
            if is_prompt_injection(user_question):
                    st.error(
                        "Potential prompt injection detected."
                    )

                    st.stop()
            # ------------------------------
            # USER MESSAGE
            # ------------------------------

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": user_question
                }
            )

            with st.chat_message(
                "user"
            ):

                st.markdown(
                    user_question
                )

            # ------------------------------
            # CHAT HISTORY
            # ------------------------------

            history_text = "\n".join(
                [
                    f"{msg['role']}: {msg['content']}"
                    for msg in st.session_state.messages[-6:]
                ]
            )

            # ------------------------------
            # ASSISTANT RESPONSE
            # ------------------------------

            with st.chat_message(
                "assistant"
            ):

                with st.spinner(
                    "Searching papers..."
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

                    if not validate_response_scope(answer):

                        answer = (
                            "The uploaded papers do not contain "
                            "sufficient information to answer."
                        )
                    if len(response["context"]) == 0:
                        answer = (
                            "No relevant information found "
                            "in uploaded papers."
                        )
                    else:
                        st.markdown(
                            answer
                        )

                    # --------------------------
                    # SOURCES
                    # --------------------------

                    citations = (
                        format_citations(
                            response["context"]
                        )
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
            # SAVE RESPONSE
            # ------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )