import os
import glob
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def add_metadata(doc, doc_type):
    """Add metadata to document"""
    doc.metadata["doc_type"] = doc_type
    return doc

def load_and_chunk_documents():
    """Load PDF documents and split them into chunks"""
    print("Scanning for PDF files...")
    pdf_files = glob.glob("**/*.pdf", recursive=True)
    print(f"Found PDF files: {pdf_files}")

    documents = []
    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file}")
        file_name = os.path.basename(pdf_file)
        doc_type = file_name.replace('.pdf', '').replace('.PDF', '')
        
        try:
            # Load PDF directly
            loader = PyPDFLoader(pdf_file)
            pdf_docs = loader.load()
            
            # Add metadata to each page
            docs_with_metadata = [add_metadata(doc, doc_type) for doc in pdf_docs]
            documents.extend(docs_with_metadata)
            print(f"Loaded {len(pdf_docs)} pages")
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")

    print(f"\n Total documents loaded: {len(documents)}")
    
    if not documents:
        print("No documents were loaded!")
        print(f"Current directory: {os.getcwd()}")
        return []

    # Split documents into chunks
    print("Splitting documents into chunks...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Total number of chunks: {len(chunks)}")
    print(f"Document types found: {set(doc.metadata['doc_type'] for doc in documents)}")
    return chunks

def create_vectorstore(chunks, db_name="vector_db"):
    """Create and save vectorstore from chunks"""
    print("Creating embeddings and vectorstore...")
    # Initialize embeddings
    embeddings = OpenAIEmbeddings()
    # Delete existing collection if exists
    if os.path.exists(db_name):
        print(f"Deleting existing vectorstore: {db_name}")
        Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()
    # Create new vectorstore
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=db_name
    )
    collection = vectorstore._collection
    count = collection.count()
    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"Vectorstore created successfully!")
    print(f"{count:,} vectors with {dimensions:,} dimensions stored in '{db_name}'")
    return vectorstore

def main():
    """Main function to create vectorstore"""
    print("Starting vectorstore creation process...")
    
    chunks = load_and_chunk_documents()
    
    if not chunks:
        print("No chunks created. Exiting...")
        return
    # Create vectorstore
    db_name = "vector_db"
    vectorstore = create_vectorstore(chunks, db_name)
    print(f"Process completed! Vectorstore saved to '{db_name}'")

if __name__ == "__main__":
    main()