import anthropic

from rag.config import ANTHROPIC_API_KEY, SYNTHESIS_MODEL, SYNTHESIS_MAX_TOKENS


SYSTEM_PROMPT = (
    "You are a senior machine learning expert answering ML interview questions using retrieved results (RAG). "
    "First, extract and summarize all relevant information from each retrieved result. "
    "Then synthesize them into a single coherent answer. "
    "STYLE RULES (mandatory for every output):\n"
    "Organize answers into sections. "
    "Use bullet points. "
    "Include the parent headings and headings from RETRIEVED RESULTS in parentheses after each bullet point. "
    "Provide clear, interview-level explanations with practical insights. "
    "Do not hallucinate references or invent headings; only use headings from the retrieved results. "
    "Render math in LaTeX format where needed. "
    "Do not include extra text outside REQUIRED OUTPUT FORMAT. "
    "Cover theory and application in big-tech depth."
)

OUTPUT_FORMAT_TEMPLATE = (
    "REQUIRED OUTPUT FORMAT:\n"
    "- Definition\n"
    "- Where it comes from and why it's important\n"
    "- Core formulation\n"
    "- Major topics and top 5 most relevant topics\n"
    "- Practical trade-offs and convergence considerations\n"
    "- Other relevant interview insights (not explicitly in the retrieved results)\n\n"
)


def _build_user_message(question: str, retrieval_results_str: str) -> str:
    return (
        f"QUESTION:\n{question}\n\n"
        f"{OUTPUT_FORMAT_TEMPLATE}"
        f"RETRIEVED RESULTS:\n{retrieval_results_str}"
    )


def ask(question: str, retrieval_results_str: str) -> str:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=SYNTHESIS_MODEL,
        max_tokens=SYNTHESIS_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_message(question, retrieval_results_str)}],
    )
    return message.content[0].text
