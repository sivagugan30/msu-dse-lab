import os
import streamlit as st
from llama_index.core import SimpleDirectoryReader
from pathlib import Path

# Function to Convert PDF to Markdown
def convert_pdf_to_markdown(pdf_path):
    """Converts a single PDF file to Markdown format."""
    reader = SimpleDirectoryReader(input_files=[pdf_path])
    documents = reader.load_data()  # Load full content, not just the first page
    markdown_text = "\n\n".join([doc.text for doc in documents])  # Combine all pages
    return markdown_text

# Function to Process PDF Directory
def process_pdf_directory(directory):
    """Processes all PDFs in a given directory and converts them to Markdown."""
    markdown_files = {}
    pdf_files = [f for f in os.listdir(directory) if f.endswith(".pdf")]

    if not pdf_files:
        return None  # No PDFs found

    reader = SimpleDirectoryReader(directory, recursive=True)  # Ensure it reads all pages
    documents = reader.load_data()

    for doc in documents:
        file_name = Path(doc.metadata['file_path']).stem + ".md"
        if file_name in markdown_files:
            markdown_files[file_name] += "\n\n" + doc.text  # Append text if multiple chunks
        else:
            markdown_files[file_name] = doc.text

    return markdown_files

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["📄 PDF to Markdown Converter", "📚 Document Embedding"])

# Page 1: PDF to Markdown Converter
if page == "📄 PDF to Markdown Converter":
    st.title("📄 PDF to Markdown Converter")

    option = st.radio("Select input type:", ["Upload PDF", "Directory Path"])

    if option == "Upload PDF":
        uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

        if uploaded_file:
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())

            markdown_text = convert_pdf_to_markdown(temp_path)
            st.text_area("Converted Markdown", markdown_text, height=300)

            st.download_button(
                "Download Markdown File",
                markdown_text,
                file_name=uploaded_file.name.replace(".pdf", ".md"),
                mime="text/markdown"
            )

            os.remove(temp_path)  # Cleanup

    elif option == "Directory Path":
        directory_path = st.text_input("Enter directory path containing PDFs")

        if directory_path and os.path.isdir(directory_path):
            markdown_files = process_pdf_directory(directory_path)

            if markdown_files:
                for filename, content in markdown_files.items():
                    st.subheader(filename)
                    st.text_area(f"Content of {filename}", content, height=200)
                    st.download_button(
                        f"Download {filename}",
                        content,
                        file_name=filename,
                        mime="text/markdown"
                    )
            else:
                st.warning("No PDFs found in the directory.")

# Page 2: Document Embedding Page
elif page == "📚 Document Embedding":

    st.title("📚 Document Embedding Page")
    st.markdown("""
        Upload documents to create a custom knowledge base for the chatbot.  
        **NOTE:** You can either add documents to an **existing collection** or create a **new collection**.
    """)

    with st.form("document_input"):
        # Upload Documents
        document = st.file_uploader(
            "Upload Documents", type=['pdf', 'txt'], help=".pdf or .txt file", accept_multiple_files=True
        )

        # Model Selection Dropdown
        embedding_model = st.selectbox(
            "Select Embedding Model",
            options=[
                "text-embedding-3-small (256D) - Cost-efficient",
                "text-embedding-ada-002 (1536D) - Balanced choice",
                "text-embedding-3-large (3072D) - High performance"
            ],
            help="Choose an embedding model for document processing."
        )

        # Document Collection Selection
        existing_collections = ["Phoenix 1","Phoenix 2","Phoenix 3"]

        collection_name = st.selectbox(
            "Select Existing Collection or Create New",
            options=["<New Collection>"] + existing_collections
        )

        # Chunking Metrics
        row = st.columns(2)
        with row[0]:
            chunk_size = st.number_input("Chunk Size", value=200, min_value=50, step=50,
                                         help="Defines the number of characters per chunk")
        with row[1]:
            chunk_overlap = st.number_input("Chunk Overlap", value=10, min_value=0, step=5,
                                            help="Should be lower than chunk size")

        save_button = st.form_submit_button("Save Document")

        # Display Uploaded Files
        if save_button and document:
            st.success(
                f"✅ Saved {len(document)} documents in collection: {collection_name} using {embedding_model} | Chunk Size: {chunk_size} | Overlap: {chunk_overlap}")


"""

streamlit run /Users/sivaguganjayachandran/PycharmProjects/SSL/streamlit_app_03.py

"""






