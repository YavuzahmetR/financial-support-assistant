"""Separate operator guidance from editable customer-facing replies."""

from typing import Any


def present_result(result: dict[str, Any]) -> dict[str, Any]:
    """Keep internal failures out of customer text, including on exception routes."""
    gap = result["decision"]["information_gap"]
    completed = result["generation_status"] == "completed"
    draft = result.get("draft")
    reply = draft.get("draft", "") if completed and isinstance(draft, dict) else ""
    origin = "Model draft" if reply else "No customer reply prepared"
    titles = {
        "none": "General guidance prepared",
        "customer_detail": "Ask the customer for more details",
        "unavailable_record": "An authorized worker must check the record",
        "not_covered": "Refer this request to the relevant support team",
        "review_required": "The question or automatic route needs clarification",
    }
    actions = {
        "none": "Review the draft against its policy source, then edit or copy it.",
        "customer_detail": (
            "Ask the customer for the details listed below, then add their reply "
            "to the original question and generate again. Do not guess the cause."
        ),
        "unavailable_record": (
            "Check the listed record in an authorized support system, or refer "
            "the request to someone with access. Names and customer numbers do "
            "not give this demo access to records."
        ),
        "not_covered": (
            "Use your approved support process to refer the request. "
            "The supplied policies cannot support an answer to this question."
        ),
        "review_required": (
            "Ask the customer to clarify their request. If it is already clear, "
            "a support worker should choose the appropriate route manually."
        ),
    }
    if not reply:
        suggestions = {
            "unavailable_record": (
                "I cannot check your account or order records in this demo. "
                "An authorized support worker needs to check the relevant record."
            ),
            "not_covered": (
                "This demo's supplied policies do not cover your request. "
                "Please contact the relevant support team."
            ),
            "review_required": (
                "Could you rephrase your question and explain what you need help "
                "with? Please omit card numbers, PINs, passwords and verification "
                "codes."
            ),
        }
        reply = suggestions.get(gap, "")
        if reply:
            origin = "System suggestion"
        elif result["decision"]["selected_sources"]:
            # Keep the actual policy visible when a draft fails. This text is a
            # reference for worker editing, never a successful LLM generation.
            paragraphs = [
                source["text"] for source in result["decision"]["selected_sources"]
            ]
            reply = "\n\n".join(paragraphs)
            origin = "Policy reference"
    review_required = gap != "none" or not completed or result["requires_human_review"]
    title = titles[gap]
    action = actions[gap]
    if not completed and gap in {"none", "customer_detail"}:
        title = "Policy reference prepared for review"
        if gap in {"none", "customer_detail"}:
            action = (
                "Review the question and supplied policies, then write a customer "
                "reply by editing the policy reference below, or retry when the model "
                "is ready." + (actions[gap] if gap == "customer_detail" else "")
            ).strip()
    return {
        "title": title,
        "action": action,
        "reply": reply,
        "origin": origin,
        "review_required": review_required,
    }
