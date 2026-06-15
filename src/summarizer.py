from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )


MAP_PROMPT = """
You are a research paper analyst.

Analyze this section and extract use only 1 point in each:

- Main Idea
- Methodology
- Dataset
- Results
- Contributions

Section:

{chunk}
"""


REDUCE_PROMPT = """
You are a senior research scientist.

Using the following section summaries,
generate a complete research paper summary.

Include:

# Executive Summary

# Problem Statement

# Motivation

# Methodology

# Dataset

# Experimental Setup

# Results

# Contributions

# Limitations

# Future Work

# TLDR

Section Summaries:

{summaries}
"""


COMPARISON_PROMPT = """
You are an expert research reviewer.

Compare the following papers.

Provide:

# Similarities

# Differences

# Strengths

# Weaknesses

# Best Approach

# Final Comparative Analysis

Paper Summaries:

{paper_summaries}
"""


def group_documents_by_paper(documents):

    papers = defaultdict(list)

    for doc in documents:

        paper_name = doc.metadata.get(
            "paper",
            "Unknown Paper"
        )

        papers[paper_name].append(doc)

    return papers


def summarize_chunk(chunk_text):

    llm = get_llm()

    response = llm.invoke(
        MAP_PROMPT.format(
            chunk=chunk_text
        )
    )

    return response.content


def map_reduce_summary(docs):

    with ThreadPoolExecutor(max_workers=8) as executor:

        MAX_CHUNKS = 8

        docs = docs[:MAX_CHUNKS]
        chunk_summaries = list(
            executor.map(
                summarize_chunk,
                [
                    doc.page_content
                    for doc in docs
                ]
            )
        )

    combined_summary = "\n\n".join(
        chunk_summaries
    )

    llm = get_llm()

    final_response = llm.invoke(
        REDUCE_PROMPT.format(
            summaries=combined_summary
        )
    )

    return final_response.content


def summarize_single_paper(
    documents,
    selected_paper
):

    papers = group_documents_by_paper(
        documents
    )

    docs = papers[selected_paper]

    return map_reduce_summary(
        docs
    )


def summarize_all_papers(
    documents
):

    papers = group_documents_by_paper(
        documents
    )

    summaries = {}

    for paper_name, docs in papers.items():

        summaries[paper_name] = (
            map_reduce_summary(
                docs
            )
        )

    return summaries


def compare_papers(
    documents
):

    papers = group_documents_by_paper(
        documents
    )

    summaries = []

    for paper_name, docs in papers.items():

        summary = map_reduce_summary(
            docs
        )

        summaries.append(
            f"""
PAPER: {paper_name}

SUMMARY:
{summary}
"""
        )

    llm = get_llm()

    response = llm.invoke(
        COMPARISON_PROMPT.format(
            paper_summaries="\n\n".join(
                summaries
            )
        )
    )

    return response.content