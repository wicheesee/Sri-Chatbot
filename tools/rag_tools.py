import os
import glob
from dotenv import load_dotenv
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
# from langchain.retrievers import ContextualCompressionRetriever
# from langchain.retrievers.document_compressors import CrossEncoderReranker
# from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from models.document_models import DocumentSearchArgs
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

_retriever = None
_initialized = False

# HELPER FUNCTIONS
def _load_documents_for_bm25():
    """Load documents for BM25 retriever"""
    try:
        pdf_files = glob.glob("**/*.pdf", recursive=True)
        documents = []
        for pdf_file in pdf_files:
            file_name = os.path.basename(pdf_file)
            doc_type = file_name.replace('.pdf', '').replace('.PDF', '')
            
            loader = PyPDFLoader(pdf_file)
            pdf_docs = loader.load()
            for doc in pdf_docs:
                doc.metadata["doc_type"] = doc_type
                documents.append(doc)
        
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        return chunks
        
    except Exception as e:
        print(f"Error loading documents for BM25: {e}")
        return []

def _initialize_retriever():
    """Initialize retriever system (lazy loading)"""
    global _retriever, _initialized
    
    if _initialized:
        return _retriever
    
    db_name = os.path.join(os.path.dirname(__file__), "..", "vector_db")
    db_name = os.path.abspath(db_name)

    if not os.path.exists(db_name):
        print(f"Vectorstore '{db_name}' not found!")
        print("Please run 'create_vectorstore.py' first.")
        return None
    
    try:
        print("🔄 Initializing document search...")
        
        # Load vectorstore
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(persist_directory=db_name, embedding_function=embeddings)
        
        # Load documents for BM25
        documents = _load_documents_for_bm25()
        if not documents:
            print("No documents found for BM25. Using vector search only.")
            _retriever = vectorstore.as_retriever()
            _initialized = True
            return _retriever
        
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        bm25_retriever = BM25Retriever.from_documents(documents, k=10)
        
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.3, 0.7]
        )
        
        # # Setup reranker
        # reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        # compressor = CrossEncoderReranker(model=reranker_model, top_n=5)
        
        # _retriever = ContextualCompressionRetriever(
        #     base_compressor=compressor,
        #     base_retriever=ensemble_retriever
        # )
        _retriever = ensemble_retriever
        
        _initialized = True
        print("Document search initialized successfully")
        return _retriever
        
    except Exception as e:
        print(f"Error initializing retriever: {e}")
        return None


# MAIN TOOL FUNCTION
def search_documents(query: str, top_k: int = 10) -> str:
    """
    Mencari informasi dalam dokumen PDF yang telah diindeks.
    Args:
        query: Query atau pertanyaan untuk mencari dalam dokumen
        top_k: Jumlah hasil pencarian yang dikembalikan (1-10)
    Returns:
        String berisi hasil pencarian dokumen yang relevan
    """
    # Initialize retriever if needed
    retriever = _initialize_retriever()

    if retriever is None:
        return "Document search tidak tersedia. Pastikan vectorstore sudah dibuat."
    try:
        # limit top_k
        top_k = max(1, min(top_k, 10))
        
        # Lakukan pencarian
        results = retriever.get_relevant_documents(query)
        if not results:
            return f"Tidak ditemukan dokumen yang relevan untuk query: '{query}'"
        
        # Format hasil
        formatted_results = []
        for i, doc in enumerate(results[:top_k], 1):
            content = doc.page_content.strip()
            # Potong content jika terlalu panjang
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


# CREATE TOOL FOR LANGGRAPH
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