import os
import streamlit as st
import tempfile
from dotenv import load_dotenv
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory
from langchain.tools.render import format_tool_to_openai_function
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader
)
from supabase.client import Client, create_client

load_dotenv()

st.title("Agentic Document Q&A")
st.write("Upload documents, process them, and ask questions with an agent that can reason through complex queries")

# Sidebar for configuration
st.sidebar.header("Configuration")
model_option = st.sidebar.selectbox(
    "Select Model",
    ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"]
)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
k_docs = st.sidebar.slider("Number of documents to retrieve", min_value=1, max_value=5, value=2)
chunk_size = st.sidebar.slider("Chunk Size", min_value=500, max_value=3000, value=1000, step=100)
chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=0, max_value=500, value=100, step=10)

# Environment variables
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
groq_api_key = os.environ.get("GROQ_API_KEY")

# Check for required environment variables
if not all([supabase_url, supabase_key, groq_api_key]):
    st.error("Missing required environment variables. Please check your .env file.")
    st.stop()

# Initialize session state variables
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Setup Supabase client
try:
    supabase = create_client(supabase_url, supabase_key)
except Exception as e:
    st.error(f"Error connecting to Supabase: {e}")
    st.stop()

try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    st.error(f"Error loading embeddings model: {e}")
    st.stop()

# Initialize vector store
def init_vector_store():
    if st.session_state.vector_store is None:
        try:
            st.session_state.vector_store = SupabaseVectorStore(
                embedding=embeddings,
                client=supabase,
                table_name="documents",
                query_name="match_documents",
            )
        except Exception as e:
            st.error(f"Error initializing vector store: {e}")
            st.stop()
    return st.session_state.vector_store

# Function to load and process documents
def process_document(file):
    # Create a temporary file to save the uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file.name.split(".")[-1]}') as tmp_file:
        tmp_file.write(file.getvalue())
        file_path = tmp_file.name
    
    try:
        # Check file extension and use appropriate loader
        if file.name.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file.name.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        elif file.name.endswith('.csv'):
            loader = CSVLoader(file_path)
        else:  # Default to text loader
            loader = TextLoader(file_path)
        
        # Load documents
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        document_chunks = text_splitter.split_documents(documents)
        
        for chunk in document_chunks:
            chunk.metadata["source"] = file.name
            chunk.metadata["id"] = str(uuid.uuid4())
        
        vector_store = init_vector_store()
        vector_store.add_documents(document_chunks)
        
        os.unlink(file_path)
        
        return len(document_chunks)
    
    except Exception as e:
        # Clean up temp file
        os.unlink(file_path)
        raise e

# Search tool functions
def search_documents(query, k=2):
    """Search for relevant documents based on the query"""
    vector_store = init_vector_store()
    docs = vector_store.similarity_search(query, k=k)
    
    if not docs:
        return "No relevant documents found."
    
    result = ""
    for i, doc in enumerate(docs):
        result += f"\nDocument {i+1} (Source: {doc.metadata.get('source', 'Unknown')})\n"
        result += f"{doc.page_content}\n"
    
    return result

def search_documents_specific(query, source, k=2):
    """Search for documents from a specific source"""
    vector_store = init_vector_store()
    # Use metadata filtering to find documents from the specific source
    docs = vector_store.similarity_search(
        query, 
        k=k,
        filter={"source": source}
    )
    
    if not docs:
        return f"No relevant documents found from source: {source}"
    
    result = ""
    for i, doc in enumerate(docs):
        result += f"\nDocument {i+1} (Source: {doc.metadata.get('source', 'Unknown')})\n"
        result += f"{doc.page_content}\n"
    
    return result

def list_available_sources():
    """List all available document sources"""
    if not st.session_state.processed_files:
        return "No documents have been processed yet."
    
    sources = [file["name"] for file in st.session_state.processed_files]
    return "Available sources:\n" + "\n".join([f"- {source}" for source in sources])

def reformulate_query(original_query):
    """Reformulate the original query to improve retrieval performance"""
    llm = ChatGroq(
        model_name=model_option,
        groq_api_key=groq_api_key,
        temperature=0.2,
    )
    
    prompt = f"""Given the original user query: "{original_query}"
    
    Please reformulate this query to make it more effective for retrieval from a vector database. 
    Focus on extracting key entities, concepts, and using synonyms where appropriate.
    Return only the reformulated query with no additional explanation."""
    
    response = llm.invoke(prompt)
    return response.content

# Create agentic RAG system
def create_agentic_rag():
    # Initialize LLM
    llm = ChatGroq(
        model_name=model_option,
        groq_api_key=groq_api_key,
        temperature=temperature,
    )
    
    # Define tools
    tools = [
        Tool(
            name="SearchDocuments",
            func=lambda q: search_documents(q, k=k_docs),
            description="Useful for when you need to search for information in all the documents. Input should be a search query."
        ),
        Tool(
            name="SearchSpecificSource",
            func=lambda q: search_documents_specific(q.split("|||")[0], q.split("|||")[1], k=k_docs),
            description="Search for information in a specific document. Input should be in format: 'query|||source' (e.g., 'climate change|||report.pdf')."
        ),
        Tool(
            name="ListSources",
            func=list_available_sources,
            description="Lists all available document sources that have been processed."
        ),
        Tool(
            name="ReformulateQuery",
            func=reformulate_query,
            description="Reformulates the original query to improve retrieval performance. Input should be the original query."
        )
    ]
    
    # Initialize agent with tools
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        memory=st.session_state.memory,
        handle_parsing_errors=True,
    )
    
    # Enhance agent system message for better reasoning
    system_message = """You are an intelligent research assistant that helps users find information in their documents.

    Follow these steps when answering questions:
    1. Determine if you need to search for information in the documents
    2. Consider if you should reformulate the query to get better results
    3. Decide which documents to search in (all or specific sources)
    4. Analyze the retrieved information critically
    5. If the information is insufficient, try searching with different queries
    6. Synthesize information from multiple sources when appropriate
    7. Provide clear attribution to sources

    If you cannot find an answer in the documents, clearly state that you don't have enough information.
    Always show your reasoning process step by step.
    """
    
    agent.agent.prompt.messages[0].content = system_message
    
    return agent

tab1, tab2 = st.tabs(["Upload Documents", "Chat with Agent"])

with tab1:
    st.header("Upload Your Documents")
    st.write("Supported formats: PDF, DOCX, TXT, CSV")
    
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=["pdf", "docx", "txt", "csv"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Processing documents... This may take a while for large files."):
                for file in uploaded_files:
                    if file.name in [f["name"] for f in st.session_state.processed_files]:
                        st.warning(f"File {file.name} already processed. Skipping.")
                        continue
                    
                    try:
                        num_chunks = process_document(file)
                        st.session_state.processed_files.append({
                            "name": file.name,
                            "chunks": num_chunks
                        })
                        st.success(f"Successfully processed {file.name} into {num_chunks} chunks")
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {str(e)}")
    
    # Display processed files
    if st.session_state.processed_files:
        st.subheader("Processed Documents")
        for file in st.session_state.processed_files:
            st.write(f"📄 {file['name']} - {file['chunks']} chunks")

with tab2:
    st.header("Ask Questions About Your Documents")
    
    # Check if documents are processed
    if not st.session_state.processed_files:
        st.warning("Please upload and process documents first")
    else:
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Searching through documents and reasoning about your question..."):
                    message_placeholder = st.empty()
                    
                    try:
                        # Create agent with current settings
                        agent = create_agentic_rag()
                        
                        # Get response
                        response = agent.run(prompt)
                        
                        message_placeholder.markdown(response)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                    except Exception as e:
                        message_placeholder.error(f"Error: {str(e)}")
