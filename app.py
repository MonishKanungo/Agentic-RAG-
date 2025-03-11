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
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader
)
from supabase.client import Client, create_client

load_dotenv()


st.title("Document Q&A ")
st.write("Upload documents, process them, and ask questions based on your content")

# Sidebar for configuration
st.sidebar.header("Configuration")
model_option = st.sidebar.selectbox(
    "Select  Model",
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

# Create RAG chain
def create_rag_chain():
    # Initialize LLM
    llm = ChatGroq(
        model_name=model_option,
        groq_api_key=groq_api_key,
        temperature=temperature,
        max_tokens=512,
    )
    
   
    vector_store = init_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": k_docs})
    
    
    template = """Answer the following question based ONLY on the provided context. If the answer is not found in the context, simply state "I don't have enough information to answer this question."

    Question: {question}

    Context: {context}

    Answer:"""
    
    PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,
    )
    
    return qa_chain


tab1, tab2 = st.tabs(["Upload Documents", "Chat with Documents"])


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
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Searching through documents..."):
                    message_placeholder = st.empty()
                    
                    try:
                        # Create chain with current settings
                        chain = create_rag_chain()
                        
                        # Get response
                        result = chain({"query": prompt})
                        response = result["result"]
                        
                        # Display sources if available
                        if "source_documents" in result and result["source_documents"]:
                            response += "\n\n**Sources:**"
                            for i, doc in enumerate(result["source_documents"]):
                                source = doc.metadata.get('source', 'Unknown source')
                                response += f"\n{i+1}. {source}"
                        
                        message_placeholder.markdown(response)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                    except Exception as e:
                        message_placeholder.error(f"Error: {str(e)}")


