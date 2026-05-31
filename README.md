# Multi-PDF-Web-RAG-Assistant
Multi-PDF &amp; Web RAG Assistant is a Python-based AI app that enables chatting with multiple PDFs and web URLs. Built with LangChain, ChromaDB, and Google Generative AI, it retrieves relevant content and generates accurate, context-aware answers through a simple Streamlit interface.

# Overview

The Multi-Document & URL RAG Assistant is an intelligent AI-powered application developed using Python, Retrieval-Augmented Generation (RAG), and Large Language Models (LLMs). The system enables users to upload multiple PDF documents and provide website URLs as knowledge sources, allowing the application to generate context-aware, accurate, and relevant responses based on the provided content.

Unlike traditional AI chatbots that rely solely on pre-trained knowledge, this solution retrieves information directly from user-supplied documents and web content before generating responses. This approach improves accuracy, reduces hallucinations, and ensures responses remain grounded in the provided data sources.

# Features
Upload and process multiple PDF documents simultaneously.
Extract and analyze content from website URLs.
Intelligent document chunking for efficient retrieval.
Vector-based semantic search using embeddings.
Context-aware question answering powered by Google Generative AI.
Real-time conversational interface built with Streamlit.
Retrieval-Augmented Generation (RAG) architecture for enhanced response accuracy.
Scalable and modular design for future enhancements.

# Technology Stack
Backend
Python
LangChain
LangChain Community
LangChain Chroma
LangChain Google Generative AI
# Frontend
Streamlit
Database & Retrieval
Chroma Vector Database
Environment Management
Python Dotenv
Document Processing
Unstructured
# How It Works
1. Data Ingestion

Users can upload multiple PDF files and submit one or more website URLs. The application extracts textual content from both sources and prepares it for processing.

2. Text Processing

The extracted content is cleaned and divided into smaller text chunks. This improves retrieval efficiency and enables the system to identify the most relevant information for a given query.

3. Embedding Generation

Using Google Generative AI Embeddings, each text chunk is converted into a high-dimensional vector representation that captures semantic meaning.

4. Vector Storage

The generated embeddings are stored in the Chroma Vector Database, enabling fast similarity searches across all uploaded documents and web content.

5. Information Retrieval

When a user submits a question, the system performs a semantic search within the vector database to retrieve the most relevant content related to the query.

6. Response Generation

The retrieved context is passed to a Google Generative AI model through LangChain. The model generates a response based on the retrieved information, ensuring answers remain relevant to the provided documents and URLs.

# Use Cases
Research and academic document analysis
Business knowledge management
Legal and compliance document review
Technical documentation assistance
Website content exploration
Enterprise knowledge retrieval systems
# Installation
pip install -r requirements.txt
Required Dependencies
langchain
langchain-community
langchain-chroma
langchain-google-genai
streamlit
python-dotenv
unstructured
# System Diagram

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/8ba0a917-297b-4c44-a7c7-b6c8262f3272" />

# Workflow Diagram

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/4500ae46-fc66-4680-a0c4-8e96c06ea7d4" />

# Conclusion

This project demonstrates the practical implementation of a Retrieval-Augmented Generation (RAG) pipeline that combines document processing, web content extraction, semantic search, and generative AI. By leveraging LangChain, Chroma, Google Generative AI, and Streamlit, the application delivers a powerful and scalable solution for intelligent knowledge retrieval and conversational document analysis.
