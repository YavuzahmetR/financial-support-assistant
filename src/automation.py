"""Topic classification, policy retrieval, and bounded local drafting."""

import re
from dataclasses import asdict
from threading import Lock
from time import perf_counter
from typing import Any

from src.drafting import draft_from_sources, load_config
from src.embedding import encode_texts, open_encoder
from src.embedding import load_config as load_embedding_config
from src.intent import classify_vector, load_classifier
from src.retrieval import load_chunks

WORKBENCH_LOCK = Lock()

GAP_LABELS = {
    "none": "Supported general guidance",
    "customer_detail": "More context needed",
    "unavailable_record": "Personal record needed",
    "not_covered": "Outside the supplied policies",
    "review_required": "Routing uncertain",
}
TOPIC_POLICIES = {
    "card_arrival": {"card_delivery"},
    "card_delivery_estimate": {"card_delivery"},
    "activate_my_card": {"card_activation"},
    "card_not_working": {"card_damage", "card_shop_payments", "card_online_payments"},
    "contactless_not_working": {"card_shop_payments"},
    "declined_card_payment": {"card_shop_payments", "card_online_payments"},
}


def matches(pattern: str, query: str) -> bool:
    """Match English routing cues; these rules are not a trained gap classifier."""
    return re.search(pattern, query, flags=re.IGNORECASE) is not None


def personal_record_needed(query: str) -> bool:
    """Recognize requests for individual facts, rather than general instructions."""
    pattern = (
        r"\b(check|look up|confirm|verify|tell me)\b.*"
        r"\b(my account|my order|status|active|activated|shipped|dispatched)\b"
        r"|\b(is|has|was)\b.*\b(my card|my replacement|my order)\b.*"
        r"\b(active|activated|shipped|dispatched|ordered)\b"
        r"|\b(exact|actual|current|right now)\b.*"
        r"\b(date|location|status|arrival|active)\b"
        r"|\b(did|have)\s+(?:i|you)\b.*\b(order|activate)\b"
        r"|\bwhere (?:is|are)\b.*\b(my card|my shipment|my parcel)\b"
        r"|\b(?:can|could|would|will) you\b.*\b(see|check|confirm|look)\b.*"
        r"\b(active|activated|status|my account|my order)\b"
        r"|\b(?:can|could|would|will) you\b.*\b(check|confirm|verify|tell)\b.*"
        r"\b(charged|reversed|settled|paid|payment status|transaction status)\b"
    )
    # A procedural clause is not a request for the assistant to look up a record.
    # Check other clauses too: a mixed question can still ask for a personal fact.
    for clause in re.split(r"[.!?;,]|\b(?:and|or)\b", query, flags=re.IGNORECASE):
        if matches(r"^\s*how (?:do|can|should|would) i\b", clause):
            continue
        if matches(pattern, clause):
            return True
    return False


def dispatch_context_known(query: str) -> bool:
    """Recognize stated elapsed dispatch time in either word order."""
    return matches(r"\b(dispatch\w*|shipped|sent)\b", query) and matches(
        r"\b(?:[0-9]+|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve)\s+(?:business|working)\s+days\b"
        r"|\b(dispatched|shipped|sent)\s+(?:it\s+)?(?:yesterday|today)\b",
        query,
    )


def error_context_known(query: str) -> bool:
    """Detect an explicitly supplied error description, without requesting it again."""
    return matches(
        r"\berror(?: message| code)?\s*(?:is\s+|says\s+|[:=]\s*|[\"'])\S+",
        query,
    )


def load_workbench() -> dict[str, Any]:
    """Serialize model loading with query encoding and local generation."""
    with WORKBENCH_LOCK:
        return create_workbench()


def create_workbench() -> dict[str, Any]:
    """Load MPNet, trained logistic regression, and policy vectors under the lock."""
    embedding = load_embedding_config()
    session, tokenizer = open_encoder(embedding)
    classifier = load_classifier(encoder_config=embedding)
    chunks = load_chunks()
    vectors, diagnostics = encode_texts(
        [f"{chunk.section}\n{chunk.text}" for chunk in chunks],
        session,
        tokenizer,
        embedding["batch_size"],
    )
    if any(item["truncated"] for item in diagnostics):
        raise ValueError("A policy passage exceeds the encoder limit.")
    return {
        "session": session,
        "tokenizer": tokenizer,
        "classifier": classifier,
        "chunks": chunks,
        "vectors": vectors,
        "config": load_config(),
        "lock": WORKBENCH_LOCK,
    }


def route_question(
    query: str, topic: dict[str, Any], candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    """Map a confident topic to demo policies; surface context and record gaps."""
    gap = "none"
    missing: list[str] = []
    reason = "The predicted topic has a matching general demo policy."
    policies = TOPIC_POLICIES.get(topic["intent"], set())
    replacement = matches(r"\b(replac\w*|new card|spare card)\b", query)
    damaged = matches(r"\b(crack\w*|bent|damag\w*|broken)\b", query)
    replacement_record = replacement and matches(
        r"\b(did i order|have i ordered|check my order|order status)\b", query
    )
    code_missing = (
        matches(r"\b(payment|checkout)\b", query)
        and not matches(
            r"\b(login|log in|sign in|password|account verification)\b", query
        )
        and matches(r"\bcode\b", query)
        and matches(r"\b(verif\w*|confirmation|security)\b", query)
        and matches(
            r"\b(missing|not arriv\w*|not receiv\w*|hasn't arrived|never arriv\w*)\b",
            query,
        )
    )
    login_code = matches(r"\bcode\b", query) and matches(
        r"\b(login|log in|logging in|logging into|sign in|signed in|"
        r"signing in|password)\b",
        query,
    )
    explicit_policy = False
    # BANKING77 has no separate damaged-card replacement label.
    if replacement and (damaged or replacement_record) and matches(r"\bcard\b", query):
        policies = {"card_damage"}
        explicit_policy = damaged
    if code_missing:
        policies = {"card_online_payments"}
        explicit_policy = True
    online_decline = (
        matches(r"\bcard\b", query)
        and matches(r"\bpayment\b", query)
        and matches(r"\b(online|internet|website)\b", query)
        and matches(r"\b(declin\w*|reject\w*)\b", query)
    )
    contactless_failure = (
        matches(r"\b(contactless|tapping)\b", query)
        and matches(r"\bcard\b|\bchip and pin\b", query)
        and not matches(r"\b(atm|withdraw\w*|badge|door)\b", query)
        and matches(r"\b(not work\w*|fail\w*|will not work|does not work)\b", query)
    )
    if online_decline and not code_missing:
        policies = {"card_online_payments"}
        explicit_policy = True
    if contactless_failure and not code_missing and not online_decline:
        policies = {"card_shop_payments"}
        explicit_policy = True
    if explicit_policy:
        reason = "An explicit card procedure matches a supplied demo policy."
    if login_code:
        gap = "not_covered"
        reason = "Account login recovery is outside the supplied payment policies."
    elif matches(
        r"\b(fee|fees|price|costs?|free|refund\w*|compensation|interest|instalments?|installments?)\b",
        query,
    ):
        gap = "not_covered"
        reason = "These policies do not establish fees, refunds, or finance terms."
    elif topic["review_required"] and not explicit_policy:
        gap = "review_required"
        reason = "The topic classifier is uncertain; a person should choose the route."
    elif not policies:
        gap = "not_covered"
        reason = "The predicted banking topic has no policy in this demo."
    elif personal_record_needed(query):
        gap = "unavailable_record"
        records = {
            "card_delivery": "Card dispatch and shipment status record",
            "card_activation": "Account activation status record",
            "card_damage": "Replacement order record",
        }
        record_name = "Authorized card or transaction status record"
        for policy, record in records.items():
            if policies == {policy}:
                record_name = record
                break
        missing = [record_name]
        reason = "General policy cannot establish a personal account or order status."
    elif "card_delivery" in policies and matches(
        r"\b(late|overdue|still waiting|not arrived|hasn't arrived)\b", query
    ):
        gap = "customer_detail"
        missing = ["Card dispatch date or business days elapsed since dispatch"]
        reason = "The late-delivery threshold depends on dispatch, not the order date."
        if dispatch_context_known(query):
            gap, missing = "none", []
    elif "card_activation" in policies and matches(
        r"\b(error|fail\w*|unable|cannot|can't|not able)\b", query
    ):
        gap = "customer_detail"
        missing = ["Activation error message with sensitive details removed"]
        reason = "Activation-error follow-up needs the displayed error context."
        if error_context_known(query):
            gap, missing = "none", []
    elif "card_damage" in policies and damaged:
        policies = {"card_damage"}
    elif (
        not explicit_policy
        and topic["intent"] == "card_not_working"
        and not matches(
            r"\b(contactless|tap|chip|terminal|shop|online|internet|website)\b", query
        )
    ):
        policies = {"card_damage"}
        gap = "customer_detail"
        missing = [
            "Payment channel and error message with sensitive details removed",
            "Whether the card is new and whether it appears frozen in the app",
        ]
        reason = "A vague card problem does not identify the troubleshooting procedure."
    elif (
        not code_missing
        and not contactless_failure
        and (
            online_decline
            or topic["intent"] in {"card_not_working", "declined_card_payment"}
        )
    ):
        gap = "customer_detail"
        missing = []
        if not matches(
            r"\b(online|internet|website|shop|terminal|chip|contactless)\b", query
        ):
            missing.append("Payment channel: online, chip and PIN, or contactless")
        if not error_context_known(query):
            missing.append("Displayed error message with sensitive details removed")
        if matches(r"\b(online|internet|website)\b", query) and not matches(
            r"\b(verification|3d secure|confirmation code)\b", query
        ):
            missing.append("Whether the merchant requested payment verification")
        if matches(r"\b(shop|terminal|chip)\b", query) and not matches(
            r"\b(another terminal|other terminals|different terminals)\b", query
        ):
            missing.append("Whether the same problem occurs at another terminal")
        if not missing:
            gap = "none"
        reason = "A decline needs context; policy does not prove its cause."

    # Prefer the applicable procedure over other sections sharing the topic.
    sections = set()
    if "card_activation" in policies:
        sections.add(
            "activation-error"
            if matches(r"\b(error|fail\w*|unable|cannot|can't|not able)\b", query)
            else "activate-a-physical-card"
        )
    if "card_delivery" in policies:
        sections.add(
            "tracking-and-late-delivery"
            if gap == "unavailable_record"
            or matches(
                r"\b(track\w*|shipment status|delivery page|late|overdue|"
                r"not arrived|hasn't arrived)\b",
                query,
            )
            else "delivery-estimate"
        )
    if "card_damage" in policies:
        sections.add(
            "damaged-card-replacement"
            if damaged or replacement_record
            else "card-problem-details-needed"
        )
    if "card_shop_payments" in policies and matches(r"\b(contactless|tap\w*)\b", query):
        policies = {"card_shop_payments"}
        sections = {"contactless-payment-not-working"}
    elif "card_online_payments" in policies and (
        code_missing or matches(r"\b(online|internet|website)\b", query)
    ):
        policies = {"card_online_payments"}
        sections = {
            "payment-verification-code-missing"
            if code_missing
            else "online-payment-declined"
        }
    elif "card_shop_payments" in policies and matches(
        r"\b(chip|terminal|shop)\b", query
    ):
        policies = {"card_shop_payments"}
        sections = {"chip-and-pin-payment-declined"}

    selected = [
        item
        for item in candidates
        if item["source_path"].rsplit("/", 1)[-1].removesuffix(".md") in policies
        and (not sections or item["chunk_id"].split("::", 1)[1] in sections)
    ]
    if gap in {"not_covered", "review_required"}:
        selected = []
    elif not selected:
        gap, missing = "review_required", []
        reason = "The retrieved candidates do not match the predicted policy topic."
    return {
        "information_gap": gap,
        "source_ids": [item["chunk_id"] for item in selected],
        "missing_information": missing,
        "reason": reason,
        "selected_sources": selected,
    }


def analyze_question(query: str, workbench: dict[str, Any]) -> dict[str, Any]:
    """Embed once for both classification and source retrieval, without an LLM."""
    query = query.strip()
    if not 1 <= len(query) <= 1000:
        raise ValueError("Enter an English question of 1 to 1,000 characters.")
    with workbench["lock"]:
        vector, diagnostics = encode_texts(
            [query], workbench["session"], workbench["tokenizer"], 1
        )
    if diagnostics[0]["truncated"]:
        raise ValueError("Shorten the question; it exceeds the encoder limit.")
    topic = classify_vector(vector[0], workbench["classifier"])
    scores = (workbench["vectors"] @ vector.T).ravel()
    chunks = workbench["chunks"]
    ranking = sorted(
        range(len(chunks)), key=lambda i: (-float(scores[i]), chunks[i].chunk_id)
    )
    candidates = [
        {**asdict(chunks[i]), "score": float(scores[i])}
        for i in ranking
        if scores[i] > 0
    ]
    decision = route_question(query, topic, candidates)
    displayed = candidates[:3]
    displayed_ids = {item["chunk_id"] for item in displayed}
    for source in decision["selected_sources"]:
        if source["chunk_id"] not in displayed_ids:
            displayed.append(source)
    return {
        "query": query,
        "topic": topic,
        "sources": displayed,
        "decision": decision,
    }


def run_question(query: str, workbench: dict[str, Any]) -> dict[str, Any]:
    """Automate the demo; uncovered or uncertain routes make no model call."""
    started = perf_counter()
    result = analyze_question(query, workbench)
    decision = result["decision"]
    gap = decision["information_gap"]
    result.update(
        model_calls_attempted=0,
        requires_human_review=True,
        generation_status="skipped",
        draft=None,
    )
    if gap == "unavailable_record":
        result["message"] = (
            "An authorized worker must check the unavailable personal record."
        )
    elif gap in {"not_covered", "review_required"}:
        result["message"] = (
            "The supplied demo policies do not cover this request. "
            "A support worker should review it."
            if gap == "not_covered"
            else "The automatic route is uncertain. "
            "A support worker should review this request."
        )
    else:
        if not workbench["lock"].acquire(blocking=False):
            raise RuntimeError("Another request is running. Please wait.")
        result["model_calls_attempted"] = 1
        try:
            call = draft_from_sources(
                result["query"],
                decision["selected_sources"],
                workbench["config"],
                {
                    key: value
                    for key, value in decision.items()
                    if key != "selected_sources"
                },
            )
            result.update(call)
            # A phrase check is not a factual review. A worker reviews every reply.
            result["requires_human_review"] = True
            if call["generation_status"] != "completed":
                result["message"] = (
                    "The model did not return a valid answer. "
                    "A support worker should review this request."
                )
        except RuntimeError as error:
            result.update(
                generation_status="request_failed",
                validation_error=str(error),
                requires_human_review=True,
            )
            result["message"] = (
                "The local draft model is unavailable. "
                "A support worker should review this request."
            )
        finally:
            workbench["lock"].release()
    result["wall_seconds"] = round(perf_counter() - started, 3)
    return result
