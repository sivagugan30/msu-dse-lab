# msu-dse-lab

https://msu-dse-tutor-design-system-rag.streamlit.app/


RAG Chatbot Application for MSU 

This application allows users to convert PDF documents into Markdown format, embed documents into a custom knowledge base, and interact with a chatbot that utilizes Retrieval-Augmented Generation (RAG) techniques to answer queries based on indexed documents.

## Features

### 1. PDF to Markdown Converter
- **Upload PDF:** Users can upload PDF files which are then converted into Markdown format.
- **Directory Path Processing:** Users can also provide a directory path containing multiple PDFs, which are processed into individual Markdown files and available for download as a ZIP archive.

### 2. Document Embedding
- Upload PDF or TXT documents to create a custom knowledge base for the chatbot.
- Choose an embedding model (e.g., text-embedding-ada-002) for processing documents.
- Optionally add documents to an existing collection or create a new collection.
- Customize chunk size and overlap for optimal document splitting.

### 3. RAG Chatbot
- Interact with a chatbot powered by Retrieval-Augmented Generation (RAG).
- Select a document collection to query.
- Adjust various parameters like the number of top sources to view, diversity in search results, and whether to output similarity scores.
- Ask questions about the indexed documents, and the chatbot will generate context-based answers.
