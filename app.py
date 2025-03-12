import os
import streamlit as st
import tempfile
import uuid
from dotenv import load_dotenv
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
    CSVLoader,
)
from supabase.client import Client, create_client
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.memory import ConversationBufferMemory

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "query_history" not in st.session_state:
    st.session_state.query_history = []
model_options = ["llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"]
model_option = st.sidebar.selectbox("Select Model", model_options)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
chunk_size = st.sidebar.slider("Chunk Size", 500, 2000, 1000)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 500, 200)
k_docs = st.sidebar.slider("Number of documents to retrieve", 1, 10, 4)

# Advanced options
with st.sidebar.expander("Advanced Settings"):
    enable_debug = st.checkbox("Enable Debug Mode", value=False)
    reasoning_depth = st.select_slider(
        "Reasoning Depth",
        options=["Basic", "Standard", "Deep"],
        value="Standard"
    )
    max_tokens = st.number_input("Max Tokens", min_value=256, max_value=8192, value=1024)

def check_api_keys():
    missing_keys = []
    if not GROQ_API_KEY:
        missing_keys.append("GROQ_API_KEY")
    if not SUPABASE_URL:
        missing_keys.append("SUPABASE_URL")
    if not SUPABASE_KEY:
        missing_keys.append("SUPABASE_KEY")
    
    if missing_keys:
        st.error(f"Missing environment variables: {', '.join(missing_keys)}. Please add them to your .env file.")
        return False
    return True

def init_supabase_client():
    if not check_api_keys():
        return None
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        st.error(f"Error connecting to Supabase: {str(e)}")
        return None
        
def init_vector_store():
    supabase_client = init_supabase_client()
    if not supabase_client:
        return None
    
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = SupabaseVectorStore(
            client=supabase_client,
            embedding=embeddings,
            table_name="documents",
            query_name="match_documents"
        )
        return vector_store
    except Exception as e:
        st.error(f"Error initializing vector store: {str(e)}")
        return None
        
def process_document(file):
    # Create a temporary file
    suffix = os.path.splitext(file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(file.read())
        temp_path = temp.name
    
    try:
        if suffix.lower() == ".pdf":
            loader = PyPDFLoader(temp_path)
        elif suffix.lower() == ".docx":
            loader = Docx2txtLoader(temp_path)
        elif suffix.lower() == ".txt":
            loader = TextLoader(temp_path)
        elif suffix.lower() == ".csv":
            loader = CSVLoader(temp_path)
        else:
            os.unlink(temp_path)
            raise ValueError(f"Unsupported file format: {suffix}")
        
        documents = loader.load()
        
        
        file_id = str(uuid.uuid4())
        for doc in documents:
            doc.metadata["source"] = file.name
            doc.metadata["file_id"] = file_id

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        
    
        vector_store = init_vector_store()
        if vector_store:
            vector_store.add_documents(chunks)
        else:
            raise ValueError("Vector store initialization failed")
        
        
        os.unlink(temp_path)
        
        return len(chunks)
    except Exception as e:
        
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise e

# Function to create enhanced RAG tool with reasoning
def create_rag_tool():
    if not check_api_keys():
        return None
        
    try:
        llm = ChatGroq(
            model_name=model_option,
            groq_api_key=GROQ_API_KEY,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        vector_store = init_vector_store()
        if not vector_store:
            raise ValueError("Vector store initialization failed")
            
        retriever = vector_store.as_retriever(search_kwargs={"k": k_docs})
        
  
        template = """
        # Objective
        Answer the user's question based ONLY on the provided context from their documents.
        
        # Reasoning Process
        1. First, analyze the user's question: {question}
        2. Identify the key information needed to answer this question
        3. Carefully review the retrieved context
        4. Consider what the context says and doesn't say about the question
        
        # Retrieved Context
        {context}
        
        # Instructions
        - If the answer is clearly found in the context, provide it with specific references
        - If the context contains partial information, provide what's available and acknowledge limitations
        - If the context doesn't contain relevant information, state "I don't have enough information in the documents to answer this question"
        - DO NOT use information outside the provided context
        - Cite specific sources from the documents when possible
        
        # Answer
        """
        
        PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True,
        )

        def rag_tool(query: str):
            result = qa_chain({"query": query})
            response = result["result"]
            
            # Add source information
            if "source_documents" in result and result["source_documents"]:
                sources = []
                for i, doc in enumerate(result["source_documents"]):
                    source = doc.metadata.get("source", "Unknown source")
                    sources.append(f"{i+1}. {source}")
                
                if sources:
                    response += "\n\n**Sources:**\n" + "\n".join(sources)
            
            return response

        return Tool(
            name="Document Retrieval and Analysis Tool",
            func=rag_tool,
            description="Analyzes the documents to find relevant information, reason about it, and provide well-sourced answers.",
        )
    except Exception as e:
        st.error(f"Error creating RAG tool: {str(e)}")
        return None

def create_agent():
    if not check_api_keys():
        return None
        
    try:
        rag_tool = create_rag_tool()
        if not rag_tool:
            raise ValueError("RAG tool initialization failed")
            
        tools = [rag_tool]
        
    
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        
        llm = ChatGroq(
            model_name=model_option,
            groq_api_key=GROQ_API_KEY,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
     
        system_message = f"""
        You are an advanced document analysis assistant that helps users understand their documents.
        
        Follow this reasoning process for each query:
        1. UNDERSTAND: First, understand exactly what the user is asking
        2. PLAN: Determine what information you need to find in the documents
        3. RETRIEVE: Use your document retrieval tool to get relevant information
        4. ANALYZE: Carefully examine the retrieved information
        5. SYNTHESIZE: Combine information from multiple sources if needed
        6. REASON: Draw logical conclusions based on the evidence
        7. RESPOND: Provide a clear, concise answer with source references
        
        Reasoning depth: {reasoning_depth}
        
        Important guidelines:
        - Only use information found in the user's documents
        - Cite specific sources
        - Acknowledge limitations in the available information
        - Be truthful and accurate
        """
        
        agent_chain = initialize_agent(
            tools,
            llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=enable_debug,
            memory=memory,
            agent_kwargs={
                "system_message": system_message
            }
        )
        
        return agent_chain
    except Exception as e:
        st.error(f"Error creating agent: {str(e)}")
        return None
        
def display_reasoning(query):
    st.write("🧠 **Reasoning Process**")
    
    reasoning_steps = [
        "**Understanding Query**: Analyzing what information is being requested",
        "**Planning Retrieval**: Determining key terms and concepts to search for",
        "**Analyzing Documents**: Examining retrieved information for relevance",
        "**Synthesizing Answer**: Combining information from multiple sources"
    ]
    
    with st.expander("View Reasoning Steps", expanded=False):
        for step in reasoning_steps:
            st.write(step)
            st.progress(100)
            
# Streamlit UI
st.title(" Document Q&A Assistant")

tab1, tab2, tab3 = st.tabs(["Upload Documents", "Chat with Documents", "Query History"])

with tab1:
    st.header("Upload Your Documents")
    st.write("Supported formats: PDF, DOCX, TXT, CSV")
    
 
    if not check_api_keys():
        st.warning("Please add API keys to your .env file before proceeding")
    
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
                            "chunks": num_chunks,
                            "processed_at": uuid.uuid4()  # To force refresh when reprocessed
                        })
                        st.success(f"Successfully processed {file.name} into {num_chunks} chunks")
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {str(e)}")
    
    # Display processed files
    if st.session_state.processed_files:
        st.subheader("Processed Documents")
        col1, col2 = st.columns([3, 1])
        with col1:
            for file in st.session_state.processed_files:
                st.write(f"📄 {file['name']} - {file['chunks']} chunks")
        
        with col2:
            if st.button("Clear All Documents"):
                # This doesn't delete from Supabase - would need additional implementation
                st.session_state.processed_files = []
                st.experimental_rerun()

with tab2:
    st.header("Ask Questions About Your Documents")
    if not st.session_state.processed_files:
        st.warning("Please upload and process documents first")
    else:
        # Initialize or refresh agent when needed
        agent_needs_refresh = False
        if "agent" not in st.session_state:
            agent_needs_refresh = True
        elif st.sidebar.button("Refresh Agent"):
            agent_needs_refresh = True
            
        if agent_needs_refresh:
            with st.spinner("Initializing Q&A agent..."):
                st.session_state.agent = create_agent()
                if not st.session_state.agent:
                    st.error("Failed to initialize agent. Check the logs for errors.")

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about your documents"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            with st.chat_message("assistant"):
                if enable_debug:
                    display_reasoning(prompt)
                    
                with st.spinner("Analyzing documents and reasoning..."):
                    try:
                        if st.session_state.agent:
                            response = st.session_state.agent.run(input=prompt)
                            st.write(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            
                            # Add to query history
                            st.session_state.query_history.append({
                                "query": prompt,
                                "response": response,
                                "timestamp": uuid.uuid4()  # Simple timestamp substitute
                            })
                        else:
                            st.error("Agent not initialized. Please refresh the page and try again.")
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": f"I encountered an error: {error_msg}"})

with tab3:
    st.header("Query History")
    if not st.session_state.query_history:
        st.info("No queries yet. Ask questions in the Chat tab to build history.")
    else:
        for i, item in enumerate(reversed(st.session_state.query_history)):
            with st.expander(f"Query: {item['query'][:50]}{'...' if len(item['query']) > 50 else ''}", expanded=(i == 0)):
                st.write("**Your question:**")
                st.write(item["query"])
                st.write("**Assistant's answer:**")
                st.write(item["response"])
                
                if st.button(f"Ask this again", key=f"reuse_{i}"):
                   
                    st.session_state.reuse_query = item["query"]
                    st.experimental_rerun()
