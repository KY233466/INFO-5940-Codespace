import streamlit as st
import os
from openai import OpenAI
from os import environ

from extract_text import extract_text

client = OpenAI(
    api_key=os.environ["API_KEY"],
    base_url="https://api.ai.it.cornell.edu",
)

st.title("📝 File Q&A with OpenAI")
uploaded_file = st.file_uploader("Upload an article", type=("txt", "pdf"))

question = st.chat_input(
    "Ask something about the article",
    disabled=not uploaded_file,
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Ask something about the article"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if question and uploaded_file:
    # Read the content of the uploaded file
    try:
        file_content = extract_text(uploaded_file)
    finally:
        # Reset the file pointer so the widget doesn't get "stuck" if you reuse it
        uploaded_file.seek(0)

    print(file_content)

    if not file_content.strip():
        st.error("I couldn't extract any text from that file. Try another file?")
    else:
        # Append the user's question to the messages
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model="openai.gpt-4o",
                messages=[
                    {"role": "system", "content": f"Here's the content of the file:\n\n{file_content}"},
                    *st.session_state.messages
                ],
                stream=True
            )
            response = st.write_stream(stream)

        # Append the assistant's response to the messages
        st.session_state.messages.append({"role": "assistant", "content": response})


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