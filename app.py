# Importing essentials
import streamlit as st
import asyncio
import nest_asyncio
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory

# ------------------- Fix Async Issues -------------------
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
nest_asyncio.apply()

load_dotenv()

# ------------------- Streamlit Page Setup -------------------
st.set_page_config(page_title='URL + PDF + HTML Chatbot 🤖', page_icon='🌐', layout="wide")

# ------------------- Custom Theme -------------------
st.markdown("""
    <style>
        body {
            background: linear-gradient(to right, #141e30, #243b55);
            color: #ffffff;
            font-family: 'Segoe UI', sans-serif;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🌐 DataCrumbs Chat Assistant")
st.markdown("Chat with **Website + PDF + HTML + Custom Links**. Your chat history will be saved in this session.")

# ------------------- Sidebar -------------------
st.sidebar.header("📂 Upload & Train")

# PDF Upload
uploaded_pdf = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
train_pdf_btn = st.sidebar.button("📚 Train on PDF")

# HTML Upload
uploaded_html = st.sidebar.file_uploader("Upload an HTML File", type=["html"])
train_html_btn = st.sidebar.button("🌐 Train on HTML")

# Add Custom Link
custom_link = st.sidebar.text_input("Paste a Link to Train")
train_link_btn = st.sidebar.button("🔗 Train on Link")

# ------------------- Load Default Website Docs -------------------
urls = [
    "https://datacrumbs.org/",
    "https://datacrumbs.org/genai-bootcamp/",
    "https://datacrumbs.org/data-science-master/",
    "https://datacrumbs.org/ultimate-python-bootcamp-2/",
    "https://datacrumbs.org/our-courses/",
    "https://datacrumbs.org/internship/",
    "https://datacrumbs.org/contact/"
]
loader = UnstructuredURLLoader(urls=urls)
data = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
docs = text_splitter.split_documents(data)

# Embeddings & Vector Store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

# ------------------- Train on PDF -------------------
if train_pdf_btn and uploaded_pdf:
    with st.spinner("Processing PDF..."):
        temp_path = "temp.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_pdf.read())

        pdf_loader = PyPDFLoader(temp_path)
        pdf_data = pdf_loader.load()
        pdf_docs = text_splitter.split_documents(pdf_data)

        if pdf_docs:  # ✅ only add if non-empty
            vectorstore.add_documents(pdf_docs)
            st.sidebar.success("✅ PDF trained successfully!")
        else:
            st.sidebar.error("❌ No text found in PDF.")

# ------------------- Train on HTML -------------------
if train_html_btn and uploaded_html:
    with st.spinner("Processing HTML..."):
        temp_path = "temp.html"
        with open(temp_path, "wb") as f:
            f.write(uploaded_html.read())

        html_loader = UnstructuredHTMLLoader(temp_path)
        html_data = html_loader.load()
        html_docs = text_splitter.split_documents(html_data)

        if html_docs:  # ✅ only add if non-empty
            vectorstore.add_documents(html_docs)
            st.sidebar.success("✅ HTML trained successfully!")
        else:
            st.sidebar.error("❌ No text found in HTML file.")

# ------------------- Train on Custom Link -------------------
if train_link_btn and custom_link:
    with st.spinner("Fetching & Training on Link..."):
        url_loader = UnstructuredURLLoader(urls=[custom_link])
        link_data = url_loader.load()
        link_docs = text_splitter.split_documents(link_data)

        if link_docs:  # ✅ only add if non-empty
            vectorstore.add_documents(link_docs)
            st.sidebar.success("✅ Link trained successfully!")
        else:
            st.sidebar.error("❌ No text extracted from this link.")

# ------------------- Retriever -------------------
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 7})

# ------------------- LLM -------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, max_tokens=500)

# ------------------- Chat Prompt -------------------
system_prompt = (
    """ You are an assistant responsible for answering queries of students
    interested in learning about the website. If you don't know the answer,
    just reply: 'I don't know the answer regarding it but you can contact the organization by Sending email: info@datacrumbs.org'
    Keep answers concise and clear for students using bullets.
    (max 4-5 sentences).
    
    {context}
    """
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# ------------------- Session Memory -------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

memory = ConversationBufferMemory()

# ------------------- Chat Interface -------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input Bar at Bottom
if query := st.chat_input("💬 Ask me anything..."):
    # Save user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # RAG Chain
    question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    with st.spinner("Thinking..."):
        response = rag_chain.invoke({"input": query})
        answer = response["answer"]

    # Save bot response
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
