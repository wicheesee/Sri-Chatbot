import os
import glob
from dotenv import load_dotenv
import gradio as gr
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

# Configuration
MODEL = "gpt-4o-mini"
db_name = "vector_db"

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')

def add_metadata(doc, doc_type):
    """Add metadata to document"""
    doc.metadata["doc_type"] = doc_type
    return doc

def load_documents_for_bm25():
    """Load documents specifically for BM25 retriever"""
    print("Loading documents for BM25 retriever...")
    
    pdf_files = glob.glob("**/*.pdf", recursive=True)
    documents = []
    
    for pdf_file in pdf_files:
        file_name = os.path.basename(pdf_file)
        doc_type = file_name.replace('.pdf', '').replace('.PDF', '')
        
        try:
            loader = PyPDFLoader(pdf_file)
            pdf_docs = loader.load()
            docs_with_metadata = [add_metadata(doc, doc_type) for doc in pdf_docs]
            documents.extend(docs_with_metadata)
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")

    # Split for BM25
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"✓ Loaded {len(chunks)} chunks for BM25")
    return chunks

def load_vectorstore(db_name):
    """Load existing vectorstore"""
    print(f"Loading vectorstore from '{db_name}'...")
    if not os.path.exists(db_name):
        print(f"Vectorstore '{db_name}' not found!")
        print("Please run 'create_vectorstore.py' first to create the vectorstore.")
        return None
    try:
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(
            persist_directory=db_name, 
            embedding_function=embeddings
        )
        # Check if vectorstore is properly loaded
        count = vectorstore._collection.count()
        print(f"Vectorstore loaded successfully with {count:,} documents")
        return vectorstore
    except Exception as e:
        print(f"Error loading vectorstore: {e}")
        return None

def setup_retrievers(vectorstore, documents):
    """Setup hybrid retrieval system"""
    print("Setting up retrieval system...")
    # Vector retriever
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    # BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(documents, k=10)
    # Ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.3, 0.7]
    )
    # Reranker
    reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=reranker_model, top_n=8)
    # Final retriever with compression
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )
    print("Retrieval system setup complete")
    return compression_retriever

def create_conversation_chain(retriever):
    """Create conversational retrieval chain"""
    print("Setting up conversation chain...")
    
    llm = ChatOpenAI(temperature=0.7, model_name=MODEL)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=retriever, 
        memory=memory, 
        callbacks=[StdOutCallbackHandler()]
    )
    print("Conversation chain ready")
    return conversation_chain

def initialize_app():
    """Initialize the application"""
    print("Initializing application...")
    # Load vectorstore
    vectorstore = load_vectorstore(db_name)
    if vectorstore is None:
        return None
    # Load documents for BM25 (this is separate from embedding)
    documents = load_documents_for_bm25()
    # Setup retrievers
    retriever = setup_retrievers(vectorstore, documents)
    # Create conversation chain
    conversation_chain = create_conversation_chain(retriever)
    print("🎉 Application initialized successfully!")
    return conversation_chain
# Global variable for conversation chain
conversation_chain = None


def chat(question, history):
    """Chat function for Gradio interface"""
    global conversation_chain
    
    if conversation_chain is None:
        return "Application not initialized. Please check the console for errors."
    
    try:
        result = conversation_chain.invoke({"question": question})
        return result["answer"]
    except Exception as e:
        print(f"Error in chat: {e}")
        return f"Error occurred: {str(e)}"

def main():
    """Main function"""
    global conversation_chain
    # Initialize the app
    conversation_chain = initialize_app()

    if conversation_chain is None:
        print("Failed to initialize application. Please run 'create_vectorstore.py' first.")
        return
    
    # Launch Gradio interface
    print("Launching Gradio interface...")
    view = gr.ChatInterface(chat, type="messages").launch(inbrowser=True)

if __name__ == "__main__":
    main()