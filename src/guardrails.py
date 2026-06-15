INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all instructions",
    "system prompt",
    "reveal prompt",
    "developer message",
    "act as",
    "jailbreak",
    "bypass",
    "forget previous instructions"
]


def is_prompt_injection(query: str):

    query = query.lower()

    return any(
        pattern in query
        for pattern in INJECTION_PATTERNS
    )
def validate_response_scope(response):

    banned_phrases = [
        "as an ai",
        "general knowledge",
        "outside the provided context"
    ]

    response = response.lower()

    return not any(
        phrase in response
        for phrase in banned_phrases
    )