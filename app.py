import os
import streamlit as st
# from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.openai import OpenAI
# from llama_index.llms import OpenAI
import openai

st.set_page_config(page_title="Chat with the Medical Questions Agent", page_icon="🦙", layout="centered",
                    initial_sidebar_state="auto", menu_items=None)

openai.api_key = os.environ["OPENAI_API_KEY"] if "OPENAI_API_KEY" in os.environ else st.secrets.openai_key

st.title("Chat with the Medical Agent 💬🧊")

if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Medical Health or request a summary!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing Medical Health docs – hang tight! This should take 3-5 minutes."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.2, 
                system_prompt="""You are an expert on the Medical Health and your job is to answer medical questions.
                        Assume that all questions are related to the Medical Health. Keep your answers technical and 
                        based on facts, do not hallucinate features."""))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
    # TODO - investigate other modes like ReAct

if prompt := st.chat_input("Your question"): # Get user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history