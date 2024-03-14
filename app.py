import os
import streamlit as st
# from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.openai import OpenAI
# from llama_index.llms import OpenAI
import openai
from llama_index.core.node_parser.text import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.postprocessor import SentenceTransformerRerank
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

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

        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.3, 
                system_prompt="""You are an expert on the Medical Health and your job is to answer medical questions.
                        Assume that all questions are related to the Medical Health. Keep your answers technical and 
                        based on facts, do not hallucinate features."""))

        sentence_node_parser = SentenceSplitter.from_defaults(
        paragraph_separator=r"\n(?:●|-|\s{2,}|\.\s|？|！)\n",  # Regular expression pattern for paragraph separation
        chunk_size=512,
        include_prev_next_rel=True,   # Include previous and next relationships for nodes
        include_metadata=True         # Include metadata for nodes (such as document information)
        )

        # initialize client, setting path to save data
        chroma_client = chromadb.PersistentClient(path="./chroma_db")

        # create collection
        chroma_collection = chroma_client.get_or_create_collection("tech16example")

        # assign chroma as the vector_store to the context
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection) # set up ChromaVectorStore and load in data

        #The storage context container is a utility container for storing nodes, indices, and vectors.
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        base_nodes = sentence_node_parser.get_nodes_from_documents(docs)# Get nodes from the cleaned documents using the sentence_node_parser

        # OpenAI's text embeddings measure the relatedness of text strings. 
        # An embedding is a vector (list) of floating point numbers. The distance between two vectors measures their relatedness. 
        # Small distances suggest high relatedness and large distances suggest low relatedness.
        embed_model = OpenAIEmbedding()

        index = VectorStoreIndex(docs, base_nodes, embed_model=embed_model, service_context=service_context, storage_context = storage_context)

        return index

index = load_data()

#Post Processing - single sentences are replaced with a window containing the surrounding sentences
#using the MetadataReplacementPostProcessor to replace the sentence in each node with it’s surrounding context.
# Create a MetadataReplacementPostProcessor instance
postproc = MetadataReplacementPostProcessor(
    target_metadata_key="window"  # Specifies the target metadata key for replacement
)

#Reranking - re-order nodes, and returns the top N nodes
# Create a SentenceTransformerRerank instance
rerank = SentenceTransformerRerank(
    top_n=3, model="BAAI/bge-reranker-base"
)

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="react", verbose=True,
        similarity_top_k=10,                   # Retrieve the top 10 most similar results
    node_postprocessors=[postproc, rerank] # Apply post-processing techniques (postproc and rerank) to the retrieved nodes
)
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
