import os
import hashlib
import textwrap
import streamlit as st
from openai import OpenAI
from extract_text import extract_text

client = OpenAI(
    api_key=os.environ["API_KEY"],
    base_url="https://api.ai.it.cornell.edu",
)

st.set_page_config(page_title="📝 Multi-Doc Q&A", layout="centered")
st.title("📝 File Q&A")

# --- Session state setup ---
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Ask something about the article"}]

if "docs" not in st.session_state:
    st.session_state["docs"] = {}

uploaded_files = st.file_uploader(
    "Upload one or more documents (.txt or .pdf)",
    type=("txt", "pdf"),
    accept_multiple_files=True,
)

# --- Ingest newly uploaded files ---
def _doc_id_for(file_name: str, content: str) -> str:
    h = hashlib.sha1()
    h.update(file_name.encode("utf-8"))
    h.update(str(len(content)).encode("utf-8"))
    return h.hexdigest()[:12]

if uploaded_files:
    for uf in uploaded_files:
        # Important: each read consumes the buffer – reset afterward so Streamlit can re-use it
        content = extract_text(uf)
        uf.seek(0)

        if not content.strip():
            st.warning(f"Could not extract text from **{uf.name}** (skipping).")
            continue

        doc_id = _doc_id_for(uf.name, content)
        st.session_state["docs"][doc_id] = {"name": uf.name, "content": content}

# --- Helper: safe truncation of large context ---
def truncate_context(text: str, max_chars: int = 120_000) -> str:
    if len(text) <= max_chars:
        return text
    # Keep head and tail to retain beginnings and conclusions
    head = text[: int(max_chars * 0.6)]
    tail = text[-int(max_chars * 0.2) :]
    middle_note = "\n\n...[truncated to fit model context]...\n\n"
    return head + middle_note + tail

# --- Helper: OpenAI stream → text chunks ---
def openai_text_chunks(openai_stream):
    for ev in openai_stream:
        # Chat Completions (OpenAI 1.x) shape
        try:
            delta = ev.choices[0].delta
            if getattr(delta, "content", None):
                yield delta.content
                continue
        except Exception:
            pass
        # If your proxy uses a different event shape, ignore non-text safely.

# --- UI: Render prior conversation (show which docs were used for user messages) ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("role") == "user" and msg.get("doc_ids"):
            labels = [st.session_state["docs"][i]["name"] for i in msg["doc_ids"] if i in st.session_state["docs"]]
            if labels:
                st.caption("Used: " + " • ".join(labels))

# --- Document toggle row (only if we have docs) ---
docs_present = len(st.session_state["docs"]) > 0

selected_doc_ids = []
if docs_present:
    st.markdown("**Select which document(s) to query:**")
    doc_items = list(st.session_state["docs"].items())  # [(doc_id, {name, content}), ...]
    # Make a single-row grid of toggles
    cols = st.columns(min(4, len(doc_items)))  # up to 4 columns; rest wrap to new line
    # We collect toggles but keep state keys stable across reruns
    toggled = {}
    for idx, (doc_id, meta) in enumerate(doc_items):
        col = cols[idx % len(cols)]
        with col:
            key = f"toggle_{doc_id}"
            # default to True when first created
            if key not in st.session_state:
                st.session_state[key] = True
            toggled[doc_id] = st.toggle(meta["name"], value=st.session_state[key], key=key)
    # Compute the selected list
    selected_doc_ids = [doc_id for doc_id, on in toggled.items() if on]

# --- Chat input (only appears AFTER at least one doc uploaded) ---
placeholder = "Upload document(s) above to start." if not docs_present else "Ask something about your uploaded documents."
prompt = st.chat_input(placeholder, disabled=not docs_present)

# --- On submit: build context from selected docs, stream answer, and record selection per question ---
if prompt and docs_present:
    # Append user message (and attach which docs they selected for this question)
    st.session_state.messages.append({"role": "user", "content": prompt, "doc_ids": selected_doc_ids})

    # Build system context from the chosen docs (only those toggled ON right now)
    if not selected_doc_ids:
        st.warning("You didn't select any documents. I’ll answer without document context.")
        combined = ""
    else:
        # Delimit each doc clearly
        parts = []
        for doc_id in selected_doc_ids:
            meta = st.session_state["docs"].get(doc_id)
            if not meta:
                continue
            parts.append(
                f"--- BEGIN DOCUMENT: {meta['name']} ---\n{meta['content']}\n--- END DOCUMENT: {meta['name']} ---\n"
            )
        combined = "\n".join(parts)

    system_msg = {
        "role": "system",
        "content": textwrap.dedent(f"""
            You are a helpful assistant that answers questions strictly based on the provided documents when they are present.
            If the answer is not found in the documents, say you don't have enough info from the uploaded files.
            Here are the documents (may be truncated):
            {truncate_context(combined)}
        """).strip(),
    }

    with st.chat_message("assistant"):
        raw_stream = client.chat.completions.create(
            model="openai.gpt-4o",
            messages=[system_msg] + [
                # Only include role/content for the visible conversation
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state["messages"]
            ],
            stream=True,
        )
        response_text = st.write_stream(openai_text_chunks(raw_stream))

    st.session_state.messages.append({"role": "assistant", "content": response_text})


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