import os
import hashlib
import streamlit as st
from openai import OpenAI
from extract_text import extract_text

from rag_pipeline import build_messages, get_docs

client = OpenAI(
    api_key=os.environ["API_KEY"],
    base_url="https://api.ai.it.cornell.edu",
)

st.set_page_config(page_title="📝 Multi-Doc Q&A", layout="centered")
st.title("📝 File Q&A")

# --- Session state ---
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Ask something about the article"}]
if "docs" not in st.session_state:
    st.session_state["docs"] = {}

uploaded_files = st.file_uploader(
    "Upload one or more documents (.txt or .pdf)",
    type=("txt", "pdf"),
    accept_multiple_files=True,
)

def _doc_id_for(file_name: str, content: str) -> str:
    h = hashlib.sha1()
    h.update(file_name.encode("utf-8"))
    h.update(str(len(content)).encode("utf-8"))
    return h.hexdigest()[:12]

# Ingest uploads
if uploaded_files:
    for uf in uploaded_files:
        content = extract_text(uf)
        uf.seek(0)
        if not content.strip():
            st.warning(f"Could not extract text from **{uf.name}** (skipping).")
            continue
        doc_id = _doc_id_for(uf.name, content)
        st.session_state["docs"][doc_id] = {"name": uf.name, "content": content}

# Render history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("role") == "user" and msg.get("doc_ids"):
            labels = [st.session_state["docs"][i]["name"] for i in msg["doc_ids"] if i in st.session_state["docs"]]
            if labels:
                st.caption("Used: " + " • ".join(labels))

docs_present = len(st.session_state["docs"]) > 0

# --- Always show toggles above chat input ---
# with st.container():
selected_doc_ids = []
if docs_present:
    st.markdown("**Select which document(s) to query:**")
    items = list(st.session_state["docs"].items())
    cols = st.columns(min(4, len(items)))
    toggled = {}
    for idx, (doc_id, meta) in enumerate(items):
        col = cols[idx % len(cols)]
        with col:
            key = f"toggle_{doc_id}"
            default_on = st.session_state.get(key, True)
            toggled[doc_id] = st.toggle(meta["name"], value=default_on, key=key)
    selected_doc_ids = [doc_id for doc_id, on in toggled.items() if on]
else:
    st.markdown("*Upload documents above to enable toggles.*")

# --- Chat input sits directly below the toggles ---
placeholder = (
    "Upload document(s) above to start."
    if not docs_present
    else "Ask something about your uploaded documents."
)
prompt = st.chat_input(placeholder, disabled=not docs_present)

# # Multi-select toggles
# selected_doc_ids = []
# if docs_present:
#     st.markdown("**Select which document(s) to query:**")
#     items = list(st.session_state["docs"].items())
#     cols = st.columns(min(4, len(items)))
#     toggled = {}
#     for idx, (doc_id, meta) in enumerate(items):
#         col = cols[idx % len(cols)]
#         with col:
#             key = f"toggle_{doc_id}"
#             # IMPORTANT: don't pre-set st.session_state[key] here.
#             default_on = st.session_state.get(key, True)  # read only
#             toggled[doc_id] = st.toggle(meta["name"], value=default_on, key=key)
#     selected_doc_ids = [doc_id for doc_id, on in toggled.items() if on]

# placeholder = "Upload document(s) above to start." if not docs_present else "Ask something about your uploaded documents."

# prompt = st.chat_input(placeholder, disabled=not docs_present)

if prompt and docs_present:
    # 1) Show the *current* user message immediately
    with st.chat_message("user"):
        st.write(prompt)
        if selected_doc_ids:
            labels = [st.session_state["docs"][i]["name"] for i in selected_doc_ids if i in st.session_state["docs"]]
            if labels:
                st.caption("Used: " + " • ".join(labels))

    # 2) Persist it for future reruns
    st.session_state["messages"].append({"role": "user", "content": prompt, "doc_ids": selected_doc_ids})

    # 3) Retrieval
    named_texts = [(st.session_state["docs"][i]["name"], st.session_state["docs"][i]["content"])
                   for i in selected_doc_ids if i in st.session_state["docs"]]

    if not named_texts:
        st.warning("You didn't select any documents. I’ll answer without document context.")
        retrieved_docs = []
    else:
        retrieved_docs = get_docs(named_texts, prompt)

    messages = build_messages(prompt, retrieved_docs)

    def openai_text_chunks(openai_stream):
        for ev in openai_stream:
            try:
                delta = ev.choices[0].delta
                if getattr(delta, "content", None):
                    yield delta.content
            except Exception:
                pass

    with st.chat_message("assistant"):
        raw_stream = client.chat.completions.create(
            model="openai.gpt-4o",
            messages=messages,
            stream=True,
        )
        response_text = st.write_stream(openai_text_chunks(raw_stream))

    st.session_state["messages"].append({"role": "assistant", "content": response_text})


# import streamlit as st
# from openai import OpenAI

# client = OpenAI()

# st.set_page_config(page_title="Hello Codespaces", layout="centered")

# st.title("👋 Hello from Codespaces!")

# with open("data/important_knowledge.txt", "r") as f:
#         knowledge_base = f.read()

# if "messages" not in st.session_state:
#     st.session_state["messages"] = [{"role": "system", "content": "You are a travel agent, helping uers find events that match their interests,"},
#                                     {"role": "user", "content": "I want you to answer questions based on this knowledge base" + knowledge_base},
#                                     {"role": "assistant", "content": "Howdy"}]

# for msg in st.session_state.messages:
#     if msg["role"] != "system":
#         st.chat_message(msg["role"]).write(msg["content"])

# if prompt := st.chat_input():
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     st.chat_message("user").write(prompt)

#     with st.chat_message("assistant"):
#         stream = client.chat.completions.create(model="openai.gpt-4o", messages=st.session_state.messages, stream=True)
#         response = st.write_stream(stream)

#     st.session_state.messages.append({"role": "assistant", "content": response})