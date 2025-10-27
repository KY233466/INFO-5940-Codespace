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

# Ingest uploads and build per-doc retrievers
if uploaded_files:
    for uf in uploaded_files:
        content = extract_text(uf)
        uf.seek(0)
        if not content.strip():
            st.warning(f"Could not extract text from **{uf.name}** (skipping).")
            continue

        doc_id = _doc_id_for(uf.name, content)
        st.session_state["docs"][doc_id] = {"name": uf.name, "content": content}

        # Build a retriever once per document (on first see or if doc content changed -> new id)
        if doc_id not in st.session_state["retrievers"]:
            st.session_state["retrievers"][doc_id] = build_retriever_for_doc(
                name=uf.name,
                content=content,
            )

docs_present = len(st.session_state["docs"]) > 0

# --- Functions for toggles for choosing to engage with specific documents ---

def _group_key(doc_id: str, group_id: int) -> str:
    return f"toggle_{doc_id}_g{group_id}"

def render_toggle_group(group_id: int):
    if not docs_present:
        return
    items = list(st.session_state["docs"].items())
    if not items:
        return
    st.markdown("**Select which document(s) to query next:**")
    cols = st.columns(min(4, len(items)))
    for idx, (doc_id, meta) in enumerate(items):
        with cols[idx % len(cols)]:
            key = _group_key(doc_id, group_id)
            default_on = st.session_state.get(key, True)  # read-only default, do not pre-set
            st.toggle(meta["name"], value=default_on, key=key)

def read_selected_doc_ids_from_group(group_id: int) -> list[str]:
    if group_id is None or not docs_present:
        return []
    selected = []
    for doc_id in st.session_state["docs"].keys():
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
        st.caption("Used: " + " • ".join(labels))

def openai_text_chunks(openai_stream):
    for ev in openai_stream:
        try:
            delta = ev.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                yield content
        except Exception:
            pass

# Render history and collect assistant message indices
assistant_msg_indices = []
for i, msg in enumerate(st.session_state["messages"]):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("role") == "user" and msg.get("doc_ids"):
            render_used_labels(msg["doc_ids"])
        # After each assistant message, show the toggle group for the *next* question
        if msg["role"] == "assistant" and docs_present:
            render_toggle_group(i)
            assistant_msg_indices.append(i)

# The latest assistant message's toggles drive the next prompt
active_group_id = assistant_msg_indices[-1] if assistant_msg_indices else None
selected_doc_ids = read_selected_doc_ids_from_group(active_group_id)

# --- Chat input ---
placeholder = "Upload document(s) above to start." if not docs_present else "Ask something about your uploaded documents."
prompt = st.chat_input(placeholder, disabled=not docs_present)

if prompt and docs_present:
    with st.chat_message("user"):
        st.write(prompt)
        render_used_labels(selected_doc_ids)

    # Persist user message (including which docs were selected)
    st.session_state["messages"].append(
        {"role": "user", "content": prompt, "doc_ids": selected_doc_ids}
    )

    # Retrieval
    selected_retrievers = [
        st.session_state["retrievers"][doc_id]
        for doc_id in selected_doc_ids
        if doc_id in st.session_state["retrievers"]
    ]
    retrieved_docs = retrieve_across(selected_retrievers, prompt, k_each=10) if selected_retrievers else []

    # Build RAG messages
    messages = build_messages(prompt, retrieved_docs)

    # Stream assistant reply
    with st.chat_message("assistant"):
        raw_stream = client.chat.completions.create(
            model="openai.gpt-4o",
            messages=messages,
            stream=True,
        )
        response_text = st.write_stream(openai_text_chunks(raw_stream))

        # Render the toggle group for THIS assistant reply inline (instant UX),
        # using the index the message will get once appended:
        current_assistant_idx = len(st.session_state["messages"])
        render_toggle_group(current_assistant_idx)

    # Persist assistant reply (so history + toggles are consistent on rerun)
    st.session_state["messages"].append({"role": "assistant", "content": response_text})