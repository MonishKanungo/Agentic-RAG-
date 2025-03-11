import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import Client, create_client
from langchain_community.embeddings import HuggingFaceEmbeddings
from huggingface_hub import login

load_dotenv()


supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")

if not supabase_url or not supabase_key:
    raise ValueError(
        "Missing Supabase environment variables. Make sure SUPABASE_URL and SUPABASE_SERVICE_KEY are in your .env file."
    )


supabase: Client = create_client(supabase_url, supabase_key)


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


if not os.path.exists("documents"):
    raise FileNotFoundError("The 'documents' directory does not exist")


loader = PyPDFDirectoryLoader("documents")

documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)


vector_store = SupabaseVectorStore.from_documents(
    docs,
    embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
)

print("Documents successfully ingested into vector store using HuggingFace embeddings.")

query = "Summarize the key points of the document."
results = vector_store.similarity_search(query)
print(results)