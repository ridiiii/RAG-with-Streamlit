import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader 
import docx
import os
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Configure page settings
st.set_page_config(page_title="Document Chat Assistant", layout="wide")

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = False
if "api_key" not in st.session_state:
    st.session_state.api_key = None

def configure_gemini():
    """Configure Gemini API with the provided API key"""
    # Check if API key is already in session state
    if st.session_state.api_key:
        return st.session_state.api_key
    
    # Try to get API key from environment variable
    api_key = os.getenv("GOOGLE_API_KEY")
    
    # If not found, ask user to input it
    if not api_key:
        st.write("Please enter your Gemini API key to continue. You can get one from https://makersuite.google.com/app/apikey")
        api_key = st.text_input("Gemini API Key:", type="password")
        if not api_key:
            st.warning("Please enter your Gemini API key to continue.")
            st.stop()
    
    # Store API key in session state and configure Gemini
    st.session_state.api_key = api_key
    genai.configure(api_key=api_key)
    return api_key

def read_pdf(file):
    """Extract text from PDF file"""
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def read_docx(file):
    """Extract text from DOCX file"""
    doc = docx.Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def read_txt(file):
    """Read text from TXT file"""
    return file.getvalue().decode("utf-8")

def process_documents(uploaded_files, api_key):
    """Process uploaded documents and create vector store"""
    text_content = []
    
    for file in uploaded_files:
        if file.type == "application/pdf":
            text_content.append(read_pdf(file))
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text_content.append(read_docx(file))
        elif file.type == "text/plain":
            text_content.append(read_txt(file))
            
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )
    text_chunks = text_splitter.split_text("\n".join(text_content))
    
    # Create embeddings and vector store with explicit API key
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    
    return vector_store

def get_response(user_question, vector_store, api_key):
    """Generate response using RAG"""
    # Search for relevant documents
    docs = vector_store.similarity_search(user_question, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    
    # Create combined prompt
    combined_prompt = (
        f"Use the following context to answer the question. If you cannot find "
        f"the answer in the context, say so.\n\nContext: {context}\n\n"
        f"Question: {user_question}\n\nAnswer: "
    )
    
    # Initialize Gemini model
    model = genai.GenerativeModel('gemini-pro')
    
    # Generate response
    response = model.generate_content(combined_prompt)
    
    return response.text

def main():
    st.title("📚 Document Chat Assistant")
    
    # Add sidebar with instructions
    with st.sidebar:
        st.header("Instructions")
        st.write("""
        1. Enter your Gemini API key
        2. Upload one or more documents (PDF, DOCX, or TXT)
        3. Wait for processing to complete
        4. Ask questions about your documents in the chat
        """)
        
        # Add API key management
        if st.button("Clear API Key"):
            st.session_state.api_key = None
            st.session_state.processed_docs = False
            st.session_state.messages = []
            st.experimental_rerun()
    
    # Configure Gemini API
    api_key = configure_gemini()
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload your documents (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    
    # Process documents when uploaded
    if uploaded_files and not st.session_state.processed_docs:
        with st.spinner("Processing documents..."):
            try:
                st.session_state.vector_store = process_documents(uploaded_files, api_key)
                st.session_state.processed_docs = True
                st.success("Documents processed successfully!")
            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")
                st.stop()
    
    # Chat interface
    if st.session_state.processed_docs:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if question := st.chat_input("Ask a question about your documents"):
            # Display user message
            with st.chat_message("user"):
                st.write(question)
            st.session_state.messages.append({"role": "user", "content": question})
            
            # Generate and display response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = get_response(question, st.session_state.vector_store, api_key)
                    st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":

    main()
