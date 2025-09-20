import os
from dotenv import load_dotenv
from langchain.tools import StructuredTool
from pydantic import BaseModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_redis import RedisVectorStore
from models.document_models import DocumentSearchArgs

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL")
DB_NAME = "vector_db"

_retriever = None
_initialized = False

INDEX_CONFIG = {
    "embedding": {
        "dimension": 1536,              
        "distance_metric": "cosine"
    }
}

def _create_vector_retriever():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = RedisVectorStore.from_existing_index(
        redis_url=REDIS_URL,
        index_name=DB_NAME,
        embedding=embeddings,
        index_config=INDEX_CONFIG
    )
    return vectorstore.as_retriever(search_kwargs={"k": 10})

try:
    print("Initializing document search (vector retriever)...")
    _retriever = _create_vector_retriever()
    _initialized = True
    print("Retriever initialized successfully")
except Exception as e:
    print(f"Failed to initialize retriever: {e}")
    _retriever = None
    _initialized = False


def _initialize_retriever():
    """Return retriever if ready"""
    global _retriever, _initialized
    if not _initialized or _retriever is None:
        print("Retriever belum tersedia")
        return None
    return _retriever

def search_documents(query: str, top_k: int = 10) -> str:
    retriever = _initialize_retriever()

    if retriever is None:
        return "Document search tidak tersedia. Pastikan vectorstore sudah dibuat."

    try:
        top_k = max(1, min(top_k, 10))
        results = retriever.get_relevant_documents(query)

        if not results:
            return f"Tidak ditemukan dokumen yang relevan untuk query: '{query}'"

        formatted_results = []
        for i, doc in enumerate(results[:top_k], 1):
            content = doc.page_content.strip()
            if len(content) > 500:
                content = content[:500] + "..."
            
            doc_type = doc.metadata.get('doc_type', 'Unknown')
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            
            formatted_results.append(
                f"**Hasil {i}:**\n"
                f"**Dokumen:** {doc_type}\n"
                f"**Halaman:** {page}\n"
                f"**Konten:**\n{content}\n"
            )

        summary = (
            f"**Pencarian Dokumen Berhasil**\n"
            f"**Query:** {query}\n"
            f"**Ditemukan:** {len(formatted_results)} hasil relevan\n\n"
        )

        return summary + "\n".join(formatted_results)

    except Exception as e:
        return f"Error saat mencari dokumen: {str(e)}"

document_search_tools = [
    StructuredTool.from_function(
        func=search_documents,
        name="search_documents",
        description=(
            "Mencari informasi dalam koleksi dokumen PDF yang telah diindeks. "
            "Gunakan tool ini ketika pengguna bertanya tentang informasi spesifik "
            "yang mungkin ada dalam dokumen atau file PDF yang tersedia."
        ),
        args_schema=DocumentSearchArgs
    )
]
