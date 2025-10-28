import os
import hashlib
import streamlit as st
from openai import OpenAI
from extract_text import extract_text
from rag_pipeline import build_messages, build_retriever_for_doc, retrieve_across

client = OpenAI(
    api_key=os.environ["API_KEY"],
    base_url="https://api.ai.it.cornell.edu",
)

st.set_page_config(page_title="📝 Multi-Doc Q&A", layout="centered")
st.title("📝 File Q&A")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Ask something about the article"}
    ]
if "docs" not in st.session_state:
    st.session_state["docs"] = {}
if "retrievers" not in st.session_state:
    st.session_state["retrievers"] = {}

uploaded_files = st.file_uploader(
    "Upload one or more documents (.txt or .pdf)",
    type=("txt", "pdf"),
    accept_multiple_files=True,
)

# Keep track of multiple documents with doc id
def _doc_id_for(file_name: str, content: str) -> str:
    h = hashlib.sha1()
    h.update(file_name.encode("utf-8"))
    h.update(str(len(content)).encode("utf-8"))
    return h.hexdigest()[:12]

# Ingest uploads & build per-doc retrievers once
if uploaded_files:
    for uf in uploaded_files:
        content = extract_text(uf)
        uf.seek(0)
        if not content.strip():
            st.warning(f"Could not extract text from **{uf.name}** (skipping).")
            continue

        doc_id = _doc_id_for(uf.name, content)
        st.session_state["docs"][doc_id] = {"name": uf.name, "content": content}
        # Build a retriever once per document
        if doc_id not in st.session_state["retrievers"]:
            st.session_state["retrievers"][doc_id] = build_retriever_for_doc(
                doc_id=doc_id,
                name=uf.name,
                content=content,
            )

docs_present = len(st.session_state["docs"]) > 0

# --- Functions for toggles for choosing to engage with specific documents ---
def _group_key(doc_id: str, group_id: int) -> str:
    return f"toggle_{doc_id}_g{group_id}"

def _group_docs_key(group_id: int) -> str:
    return f"group_docs_{group_id}"

def render_single_toggle_group_for_assistant(index_of_assistant_msg: int):
    """Render ONE toggle group under the specified assistant message and remember which doc_ids it showed."""
    if not docs_present:
        return
    items = list(st.session_state["docs"].items())
    if not items:
        return

    # Remember exactly which doc_ids are part of this group
    doc_ids_in_group = [doc_id for doc_id, _ in items]
    st.session_state[_group_docs_key(index_of_assistant_msg)] = doc_ids_in_group

    st.markdown("**Select which document(s) to query next:**")
    cols = st.columns(min(4, len(items)))
    for idx, (doc_id, meta) in enumerate(items):
        with cols[idx % len(cols)]:
            key = _group_key(doc_id, index_of_assistant_msg)
            default_on = st.session_state.get(key, True)
            st.toggle(meta["name"], value=default_on, key=key)

def read_selected_doc_ids_from_group(group_id: int) -> list[str]:
    if group_id is None or not docs_present:
        return []
    scope = st.session_state.get(_group_docs_key(group_id), [])
    selected = []
    for doc_id in scope:
        if st.session_state.get(_group_key(doc_id, group_id), True):
            selected.append(doc_id)
    return selected

# show which document user choose to engage with
def render_used_labels(doc_ids: list[str]):
    labels = [
        st.session_state["docs"][d]["name"]
        for d in (doc_ids or [])
        if d in st.session_state["docs"]
    ]
    if labels:
        st.caption("Document used: " + " • ".join(labels))

def openai_text_chunks(openai_stream):
    for ev in openai_stream:
        try:
            delta = ev.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                yield content
        except Exception:
            pass

placeholder = "Upload document(s) above to start." if not docs_present else "Ask something about your uploaded documents."
prompt = st.chat_input(placeholder, disabled=not docs_present)

# --- Render history with at most ONE toggle ---
suppress_old_toggle = bool(prompt)
last_index = len(st.session_state["messages"]) - 1
last_msg = st.session_state["messages"][last_index] if last_index >= 0 else None

last_assistant_idx = None
for i in range(last_index, -1, -1):
    if st.session_state["messages"][i]["role"] == "assistant":
        last_assistant_idx = i
        break

show_toggle_for_assistant_idx = (
    last_index if (
        last_msg and last_msg["role"] == "assistant" and docs_present and not suppress_old_toggle
    ) else None
)

for i, msg in enumerate(st.session_state["messages"]):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("role") == "user" and msg.get("doc_ids"):
            labels = [
                st.session_state["docs"][d]["name"]
                for d in (msg["doc_ids"] or [])
                if d in st.session_state["docs"]
            ]
            if labels:
                st.caption("Document used: " + " • ".join(labels))

        if show_toggle_for_assistant_idx is not None and i == show_toggle_for_assistant_idx:
            # Render the single active toggle group under the last assistant message
            render_single_toggle_group_for_assistant(i)

active_group_id = last_assistant_idx
selected_doc_ids = read_selected_doc_ids_from_group(active_group_id)

if prompt and docs_present:
    # Show user message immediately
    with st.chat_message("user"):
        st.write(prompt)
        render_used_labels(selected_doc_ids)

    # Persist user message with selection
    st.session_state["messages"].append(
        {"role": "user", "content": prompt, "doc_ids": selected_doc_ids}
    )

    # Retrieve using cached per-doc retrievers
    selected_retrievers = [
        st.session_state["retrievers"][doc_id]
        for doc_id in selected_doc_ids
        if doc_id in st.session_state["retrievers"]
    ]
    retrieved_docs = retrieve_across(selected_retrievers, prompt, k_each=10) if selected_retrievers else []

    # Build messages & stream assistant reply
    messages = build_messages(prompt, retrieved_docs)

    with st.chat_message("assistant"):
        raw_stream = client.chat.completions.create(
            model="openai.gpt-4o",
            messages=messages,
            stream=True,
        )
        response_text = st.write_stream(openai_text_chunks(raw_stream))
        st.warning(response_text)
        current_assistant_idx = len(st.session_state["messages"])
        render_single_toggle_group_for_assistant(current_assistant_idx)

    st.session_state["messages"].append({"role": "assistant", "content": response_text})