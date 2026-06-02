def format_citations(context_docs):

    citations = []

    seen = set()

    for doc in context_docs:

        source = doc.metadata.get(
            "paper",
            "Unknown"
        )

        page = doc.metadata.get(
            "page",
            "?"
        )

        citation = f"{source} (Page {page + 1})"

        if citation not in seen:

            citations.append(citation)

            seen.add(citation)

    return citations