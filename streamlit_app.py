"""Local automated support demo. Run: streamlit run streamlit_app.py."""

import streamlit as st

from src.automation import load_workbench, run_question
from src.presentation import present_result


@st.cache_resource(show_spinner="Loading local models...", max_entries=1)
def cached_workbench():
    """Share the fixed MPNet/classifier setup across sessions."""
    return load_workbench()


def clear_result() -> None:
    """An edited question invalidates its previous answer and review."""
    st.session_state.result = None
    st.session_state.answer = ""
    st.session_state.reviewed = False
    st.session_state.answer_edited = False


def invalidate_review() -> None:
    """Changing an exception answer requires a fresh review acknowledgment."""
    st.session_state.reviewed = False
    st.session_state.answer_edited = True


def main() -> None:
    """Show customer text separately from operator notices and next actions."""
    st.set_page_config(page_title="Demo Finance Support", layout="wide")
    st.title("Support assistant")
    st.caption("Local demo | English questions | Fictional policies")
    if st.session_state.get("workflow_version") != "policy_condition_checks":
        clear_result()
        st.session_state.workflow_version = "policy_condition_checks"
    try:
        workbench = cached_workbench()
    except (ValueError, RuntimeError, OSError) as error:
        st.error(str(error))
        st.info("Complete local setup: python -m scripts.setup_embedding")
        st.stop()

    query = st.text_area(
        "Customer question",
        key="query",
        max_chars=1000,
        placeholder="I placed the card order yesterday. When does it arrive?",
        on_change=clear_result,
    )
    if st.button("Generate answer", type="primary", disabled=not query.strip()):
        clear_result()
        try:
            with st.spinner(
                "Finding the topic, searching policies, and preparing the answer..."
            ):
                result = run_question(query, workbench)
            st.session_state.result = result
            st.session_state.answer = present_result(result)["reply"]
        except (ValueError, RuntimeError, OSError) as error:
            st.error(str(error))

    result = st.session_state.result
    if result is None:
        st.caption("One question -> automatic topic and sources -> answer preview")
        return
    view = present_result(result)
    gap = result["decision"]["information_gap"]
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.subheader("Answer preview")
        st.warning(view["title"] + " | Review needed before use")
        st.write("**Next action for the support worker**")
        st.write(view["action"])
        st.caption(result["decision"]["reason"])
        missing = result["decision"]["missing_information"]
        if missing:
            st.write(
                "Details to ask for:" if gap == "customer_detail" else "Record needed:"
            )
            for item in missing:
                st.write("- " + item)
        if st.session_state.answer_edited:
            st.caption("Response origin: Edited by support worker")
            st.caption("Initial response origin: " + view["origin"])
        else:
            st.caption("Response origin: " + view["origin"])
        if view["origin"] == "System suggestion" and not st.session_state.answer_edited:
            st.caption(
                "This is a standard fallback for review, not a model-generated answer."
            )
        if view["origin"] == "Policy reference" and not st.session_state.answer_edited:
            st.caption(
                "The model draft was not accepted. Edit this source text into a reply "
                "and review it before use."
            )
        st.text_area(
            "Customer reply | edit if needed",
            key="answer",
            height=220,
            on_change=invalidate_review,
        )
        reviewed = st.checkbox("I reviewed this customer reply", key="reviewed")
        if view["review_required"] and not reviewed:
            st.caption("Review the exception reply before using or downloading it.")
        st.caption(
            "Copy this reply into your support tool, or optionally download it as "
            "a text file. Nothing is sent automatically. If the customer provides "
            "more details, add them to the original question and generate again."
        )
        st.download_button(
            "Download customer reply (.txt)",
            st.session_state.answer,
            "support-answer.txt",
            disabled=not st.session_state.answer.strip()
            or (view["review_required"] and not reviewed),
        )

    with right:
        st.subheader("Topic and sources")
        topic = result["topic"]
        if gap == "review_required":
            st.write("**Topic not understood**")
            st.caption("The automatic route is uncertain. Ask for clarification.")
        else:
            st.write("Predicted topic: **" + topic["intent"].replace("_", " ") + "**")
        st.caption(
            "Used: selected for drafting or routing. "
            "Candidate: one of the three closest passages, but not selected. "
            "A matching procedure can also be selected from the remaining policies."
        )
        selected = set(result["decision"]["source_ids"])
        for source in result["sources"]:
            label = source["section"] + (
                " | used" if source["chunk_id"] in selected else " | candidate"
            )
            with st.expander(label):
                st.text(source["text"])
                st.caption(
                    f"{source['source_path']} | "
                    f"lines {source['line_start']}-{source['line_end']}"
                )
    st.caption(
        "Experimental routing | No account access | "
        "Nothing is sent to a customer automatically"
    )


if __name__ == "__main__":
    main()
