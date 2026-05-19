import os
import tempfile
import streamlit as st
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# ============================================
# FIX: Auto-load API key from Streamlit Secrets
# ============================================
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# Page configuration
st.set_page_config(
    page_title="Advanced RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Advanced RAG Documentation Assistant")
st.caption("Upload documents, ask questions, and get AI answers")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "current_docs" not in st.session_state:
    st.session_state.current_docs = []
if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = 500

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # ============================================
    # FIX: Auto-populate API key from secrets
    # ============================================
    default_key = os.getenv("GROQ_API_KEY", "")
    groq_api_key = st.text_input("Groq API Key", type="password", 
                                  value=default_key,
                                  help="Get free key at console.groq.com")
    
    st.divider()
    
    # Document Processing Settings
    st.subheader("🔧 Processing Settings")
    chunk_size = st.slider("Chunk Size (characters)", 200, 1000, st.session_state.chunk_size)
    st.session_state.chunk_size = chunk_size
    
    st.divider()
    
    # FILE UPLOAD SECTION
    st.header("📤 Upload Your Documents")
    st.caption("Supported: PDF, TXT, DOCX")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'txt', 'docx'],
        accept_multiple_files=True
    )
    
    # Text paste option
    st.caption("OR paste text directly:")
    pasted_text = st.text_area("Paste your text here", height=100)
    
    if uploaded_files and st.button("🔄 Process Files"):
        with st.spinner(f"Processing {len(uploaded_files)} files..."):
            all_documents = []
            
            for uploaded_file in uploaded_files:
                suffix = f".{uploaded_file.name.split('.')[-1]}"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                if uploaded_file.name.endswith('.pdf'):
                    loader = PyPDFLoader(tmp_path)
                elif uploaded_file.name.endswith('.docx'):
                    loader = UnstructuredWordDocumentLoader(tmp_path)
                else:
                    loader = TextLoader(tmp_path, encoding='utf-8')
                
                documents = loader.load()
                
                for doc in documents:
                    doc.metadata["source"] = uploaded_file.name
                
                all_documents.extend(documents)
                os.unlink(tmp_path)
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=st.session_state.chunk_size,
                chunk_overlap=50
            )
            chunks = text_splitter.split_documents(all_documents)
            
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
            st.session_state.current_docs = [f.name for f in uploaded_files]
            st.session_state.messages = []
            
            st.success(f"✅ Processed {len(uploaded_files)} files | {len(chunks)} chunks created")
            st.rerun()
    
    if pasted_text and st.button("🔄 Process Pasted Text"):
        with st.spinner("Processing text..."):
            from langchain_core.documents import Document
            doc = Document(page_content=pasted_text, metadata={"source": "pasted_text"})
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=st.session_state.chunk_size,
                chunk_overlap=50
            )
            chunks = text_splitter.split_documents([doc])
            
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
            st.session_state.current_docs = ["Pasted Text"]
            st.session_state.messages = []
            
            st.success(f"✅ Processed pasted text | {len(chunks)} chunks created")
            st.rerun()
    
    if st.session_state.current_docs:
        st.divider()
        st.subheader("📄 Active Documents")
        for doc in st.session_state.current_docs:
            st.caption(f"• {doc}")
        
        if st.button("🗑️ Clear Documents"):
            st.session_state.vector_store = None
            st.session_state.current_docs = []
            st.session_state.messages = []
            st.rerun()
    
    st.divider()
    
    # Export Chat History
    st.subheader("💾 Export")
    if st.session_state.messages:
        chat_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages])
        st.download_button(
            label="📥 Export Chat History",
            data=chat_text,
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.caption("Built with LangChain + FAISS + Groq Llama 3")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📄 Source documents"):
                for src in message["sources"]:
                    st.caption(f"- {src}")

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def get_rag_chain(api_key, vector_store):
    if not api_key or not vector_store:
        return None, None
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        groq_api_key=api_key
    )
    
    prompt_template = """You are a helpful technical assistant. Use ONLY the following context to answer the user's question.
If the context doesn't contain the answer, say "I don't have information about that in the uploaded documents."

Context: {context}

Question: {question}

Answer concisely and accurately:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    def chain(question):
        docs = retriever.invoke(question)
        context = format_docs(docs)
        formatted_prompt = prompt.format(context=context, question=question)
        response = llm.invoke(formatted_prompt)
        return response.content, docs
    
    return chain, retriever

# Chat input
if prompt := st.chat_input("Ask about your uploaded documents..."):
    if st.session_state.vector_store is None:
        st.warning("⚠️ Please upload documents first using the sidebar!")
        st.stop()
    
    if not groq_api_key:
        st.warning("⚠️ Please enter your Groq API key in the sidebar!")
        st.stop()
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching your documents..."):
            chain, _ = get_rag_chain(groq_api_key, st.session_state.vector_store)
            
            if chain is None:
                response = "Please ensure API key is set and documents are uploaded."
            else:
                response, docs = chain(prompt)
                sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))
                
                st.markdown(response)
                
                if sources:
                    with st.expander("📄 Source documents"):
                        for src in sources:
                            st.caption(f"- {src}")
    
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response,
        "sources": sources if 'sources' in locals() else None
    })