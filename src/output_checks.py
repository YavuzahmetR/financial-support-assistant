"""Bounded English output checks; these are not a factual-grounding evaluator."""

import re
from typing import Any

ALLOWED_GAPS = {
    "draft_ready": {"none"},
    "needs_information": {"customer_detail", "unavailable_record"},
    "unsupported": {"not_covered"},
}
SENSITIVE_TERMS = re.compile(
    r"\b(?:card[ -]+(?:number|details)|password|pin|cvv|cvc|"
    r"(?:account[ -]+)?(?:login|log[ -]in)[ -]+(?:credentials|details)|"
    r"account[ -]+credentials|"
    r"(?:verification|security|one[ -]time)[ -]+code)\b",
    re.IGNORECASE,
)
REQUEST_VERBS = re.compile(
    r"\b(?:provide|send|share|give|supply|submit|confirm|tell (?:me|us))\b",
    re.IGNORECASE,
)
NEGATED_VERB = re.compile(
    r"(?:\bnever|\b(?:do|must|should) not|\bdon't|\bcannot)\s+(?:ever\s+)?$",
    re.IGNORECASE,
)

# Short, explicit checks for known policy omissions. Matching words cannot prove
# that an answer is factually correct; the worker still sees the original source.
# Each item is (instruction sent to the model, phrases required in its answer).
POLICY_POINTS = {
    "activate-a-physical-card": [
        (
            "In the app, select the matching card and choose Activate.",
            [r"\bapp\b", r"\bactivat\w*\b"],
        ),
        (
            "After activation, insert the card and enter the PIN before contactless "
            "use.",
            [r"\b(insert\w*|chip)\b", r"\bpin\b", r"\bbefore\b.*\bcontactless\b"],
        ),
    ],
    "activation-error": [
        (
            "Check the matching card, update the app and retry on a stable connection.",
            [
                r"\b(card|matching|correct)\b",
                r"\bupdat\w*\b",
                r"\b(connection|internet|network)\b",
            ],
        ),
        (
            "If the error persists, ask for the error with sensitive details "
            "removed and contact support.",
            [r"\berror\b", r"\bsensitive\b", r"\bsupport\b"],
        ),
    ],
    "delivery-estimate": [
        (
            "Give an estimated 5 to 8 business days AFTER DISPATCH, not from ordering.",
            [
                r"\b5\s*(?:to|[-–])\s*8\b",
                r"\b(business|working) days\b",
                r"\bdispatch\w*\b",
                r"\b(estimat\w*|normally|typically|usually)\b",
            ],
        ),
        (
            "Weekends and public holidays do not count.",
            [r"\bweekends?\b", r"\bholidays?\b"],
        ),
    ],
    "tracking-and-late-delivery": [
        (
            "Track through the app's Cards and Delivery; a link appears only if "
            "supplied by the courier.",
            [r"\bapp\b", r"\bdelivery\b", r"\bcourier\b"],
        ),
        (
            "Only AFTER 8 business days from dispatch without arrival: check the "
            "address and contact support for review.",
            [
                r"\b8\b",
                r"\b(business|working) days\b",
                r"\bdispatch\w*\b",
                r"\baddress\b",
                r"\bsupport\b",
            ],
        ),
    ],
    "damaged-card-replacement": [
        (
            "Avoid forcing the damaged card into a terminal; choose Replace card in "
            "the app.",
            [r"\b(forc\w*|terminal)\b", r"\bapp\b", r"\breplac\w*\b"],
        ),
        (
            "Review any displayed cost BEFORE confirming; do not quote a fee or "
            "promise free replacement.",
            [
                r"\b(displayed|shown|listed)\b.*\b(cost|fee|price)\b|\b(cost|fee|price)\b.*\b(displayed|shown|listed)\b",
                r"\bbefore\b.*\bconfirm\w*\b",
            ],
        ),
    ],
    "card-problem-details-needed": [
        (
            "Ask for the payment channel and error with sensitive details removed.",
            [
                r"\b(contactless|channel)\b",
                r"\b(online|internet)\b",
                r"\berror\b",
                r"\bsensitive\b",
            ],
        ),
        (
            "Ask whether the card is new and appears frozen in the app; do not "
            "guess the cause.",
            [r"\bnew\b", r"\bfrozen\b", r"\bapp\b"],
        ),
    ],
    "chip-and-pin-payment-declined": [
        (
            "Ask for the error with sensitive details removed and whether other "
            "terminals fail.",
            [
                r"\berror\b",
                r"\bsensitive\b",
                r"\b(another|other|different)\b.*\bterminal\w*\b",
            ],
        ),
        (
            "Check activation, frozen status and available funds in the app.",
            [r"\bactivat\w*\b", r"\bfrozen\b", r"\b(funds|balance)\b"],
        ),
        (
            "After repeated PIN errors, stop retrying and contact support; do not "
            "invent a cause.",
            [r"\b(repeat\w*|multiple)\b", r"\bpin\b", r"\bstop\b", r"\bsupport\b"],
        ),
    ],
    "contactless-payment-not-working": [
        (
            "After activation, use chip and PIN before trying contactless.",
            [
                r"\bactivat\w*\b",
                r"\b(chip|insert\w*)\b",
                r"\bpin\b",
                r"\bbefore\b.*\bcontactless\b",
            ],
        ),
        (
            "If taps still fail at different terminals, check the contactless "
            "setting in the app and contact support.",
            [
                r"\b(different|other|multiple)\b.*\bterminal\w*\b",
                r"\bsetting\w*\b",
                r"\bapp\b",
                r"\bsupport\b",
            ],
        ),
    ],
    "online-payment-declined": [
        (
            "Enter payment details only on a legitimate merchant checkout page, "
            "never in a support message.",
            [r"\b(legitimate|genuine)\b", r"\bcheckout\b", r"\bsupport message\b"],
        ),
        (
            "Ask for the error with sensitive details removed and whether payment "
            "verification was requested.",
            [r"\berror\b", r"\bsensitive\b", r"\bverification\b"],
        ),
        (
            "Check online payment settings, expiry and card status in the app; do "
            "not claim an individual payment succeeded or reversed.",
            [
                r"\bonline\b",
                r"\b(expiry|expiration|expires)\b",
                r"\bstatus\b",
                r"\bapp\b",
            ],
        ),
    ],
    "payment-verification-code-missing": [
        (
            "Check network signal and the phone number in the app; request a new "
            "code on the payment verification screen.",
            [
                r"\b(signal|network)\b",
                r"\bphone\b",
                r"\bapp\b",
                r"\b(new|another) (code|one)\b",
                r"\b(screen|page)\b",
            ],
        ),
        (
            "If it persists, contact support with merchant and approximate attempt "
            "time; never share the code or bypass verification.",
            [
                r"\bsupport\b",
                r"\bmerchant\b",
                r"\btime\b",
                r"\b(never|do not|don't)\b.*\bshare\b",
            ],
        ),
    ],
}


def required_policy_points(sources: list[dict[str, Any]]) -> list[str]:
    """Give the draft model the important conditions for the selected sections."""
    points = []
    for source in sources:
        section = source["chunk_id"].split("::", 1)[1]
        points.extend(text for text, _ in POLICY_POINTS.get(section, []))
    return points


def missing_policy_points(text: str, sources: list[dict[str, Any]]) -> list[str]:
    """Find known missing phrases, without claiming to evaluate semantic truth."""
    missing = []
    for source in sources:
        section = source["chunk_id"].split("::", 1)[1]
        for instruction, patterns in POLICY_POINTS.get(section, []):
            if any(
                re.search(pattern, text, re.IGNORECASE | re.DOTALL) is None
                for pattern in patterns
            ):
                missing.append(instruction)
    return missing


def requests_sensitive_information(draft: dict[str, Any]) -> bool:
    """Flag known sharing requests; legitimate PIN-at-terminal steps are allowed."""
    for item in draft["missing_information"]:
        for term in SENSITIVE_TERMS.finditer(item):
            # Mentioning a payment channel is not asking for the PIN value.
            if (
                term.group().lower() == "pin"
                and re.search(r"\bchip and\s+$", item[: term.start()], re.IGNORECASE)
                and re.match(
                    r"(?:\s*[,)/]|\s*$|\s+(?:still )?(?:payments?|works?|fails?)\b)",
                    item[term.end() :],
                    re.IGNORECASE,
                )
            ):
                continue
            return True
    clauses = re.split(r"[.!?;\n]|\bbut\b", draft["draft"], flags=re.IGNORECASE)
    for clause in clauses:
        for verb in REQUEST_VERBS.finditer(clause):
            prefix = clause[: verb.start()]
            requested = clause[verb.end() :]
            if verb.group().lower() == "confirm":
                # Confirming an app action is legitimate; only the directly
                # confirmed sensitive object is a sharing request here.
                requested = re.sub(
                    r"^\s*(?:(?:your|the|my|our|full|complete|customer's)\s+)*",
                    "",
                    requested,
                    flags=re.IGNORECASE,
                )
                if not SENSITIVE_TERMS.match(requested):
                    continue
            if NEGATED_VERB.search(prefix):
                continue
            for term in SENSITIVE_TERMS.finditer(requested):
                if (
                    term.group().lower() == "pin"
                    and re.search(
                        r"\bchip and\s+$", requested[: term.start()], re.IGNORECASE
                    )
                    and re.match(
                        r"(?:\s*[,)/]|\s*$|\s+(?:still )?"
                        r"(?:payments?|works?|fails?)\b)",
                        requested[term.end() :],
                        re.IGNORECASE,
                    )
                ):
                    continue
                return True
    return False


def validate_behavior(draft: dict[str, Any]) -> None:
    """Reject inconsistent fields and known unsafe requests, without rewriting."""
    gap = draft["information_gap"]
    if not isinstance(gap, str) or gap not in ALLOWED_GAPS[draft["status"]]:
        raise ValueError("Draft status and information_gap disagree.")
    if (
        draft["status"] in {"draft_ready", "unsupported"}
        and draft["missing_information"]
    ):
        raise ValueError(
            "Ready or unsupported drafts must not request missing details."
        )
    if draft["status"] == "needs_information" and not draft["source_ids"]:
        raise ValueError("A missing-information draft must cite its policy basis.")
    if requests_sensitive_information(draft):
        raise ValueError("A sensitive-information request pattern was detected.")
    if re.search(
        r"\berror\b.*\b(?:due to|caused by)\b"
        r"|\binsert\w*\b.*\binto the app\b"
        r"|\bapplication already selected the route\b",
        draft["draft"],
        re.IGNORECASE | re.DOTALL,
    ):
        raise ValueError(
            "An unsupported cause, action or internal instruction was detected."
        )
