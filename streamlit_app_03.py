import streamlit as st
from llama_index.core import SimpleDirectoryReader
from pathlib import Path
import zipfile
from io import BytesIO
import base64
from openai import OpenAI

from default.config import (
    document_list,
    collection_list
)

import default.custom_functions as cf

import json, os

from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

json_files = ['json/Middle and High School Chemistry/' + f for f in os.listdir('json/Middle and High School Chemistry') if f.endswith('.json')]

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
page = st.sidebar.radio("Go to:", ["PDF to Markdown Converter", "Document Embedding","Chatbot"])

# Page 1: PDF to Markdown Converter
if page == "PDF to Markdown Converter":
    st.title("PDF to Markdown Converter")

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

            os.remove(temp_path)  # Cleanup 1


            print("New commit : mar 27 ")

    elif option == "Directory Path":
        directory_path = st.text_input("Enter directory path containing PDFs")

        if directory_path and os.path.isdir(directory_path):
            markdown_files = process_pdf_directory(directory_path)

            if markdown_files:
                # Create a ZIP file in memory
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for filename, content in markdown_files.items():
                        zip_file.writestr(filename, content)

                # Offer the ZIP file for download first
                zip_buffer.seek(0)
                st.markdown("---")
                st.download_button(
                    "Download All as ZIP",
                    zip_buffer,
                    file_name= directory_path.split('/')[-1] + "(markdown).zip",
                    mime="application/zip"
                )

                # Add a newline for better spacing
                st.markdown("---")

                # Display individual markdown files with download options
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
elif page == "Document Embedding":
    import os

    st.write(os.getcwd())


    st.write(os.path.exists("json/Middle and High School_Chemistry/Evaluating3DTasks.pdf"))
    st.title("Document Embedding Page")
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
        existing_collections = ["Undergraduate_Chemistry","Middle and High School_Chemistry"]

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
                f"Saved {len(document)} documents in collection: {collection_name} using {embedding_model} | Chunk Size: {chunk_size} | Overlap: {chunk_overlap}")








if page == 'Chatbot':

    st.title("RAG Chatbot")

    tab1, tab2 = st.tabs(["Chatbot", "PDF Viewer"])

    # load the json files
    vector_dict = cf.load_json_files(json_files)

    collection_dict, doc_list  = cf.get_dict_list(vector_dict)

    collection_list = ['All'] + [*collection_dict.keys()]
    doc_list = ['All'] + doc_list

    # Fast reverse lookup dictionary (for large datasets)
    document_to_collection = {
        doc: collection
        for collection, docs in collection_dict.items()
        for doc in docs
    }

    with st.sidebar:
        selected_collection = st.selectbox(
            "Select your document collection",
            collection_list
        )

        if selected_collection != 'All':
            selected_document = st.selectbox(
                "Select your document",
                ['All'] + collection_dict[selected_collection]
            )
        else:
            selected_document = st.selectbox(
                "Select your document",
                doc_list
            )

        with st.expander("⚙️ RAG Parameters"):
            top_n = st.slider(
                "Top N sources to view:", min_value=1, max_value=10, value=3, step=1
            )
            flag_mmr = st.toggle(
                "Diversity search",
                value=True,
                help="Diversity search, i.e., 1 is the most relevant, 0 being the most diverse, 1 being the least diverse. 0.5 is a balanced state.",
            )
            diversity = st.slider(
                "Diversity parameter (lambda):",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.25,
            )
            flag_similarity_score = st.toggle(
                "Output similarity score",
                value=False,
                help="A similarity score of 1 indicates the highest level of similarity between the query and the retrieved chunk.",
            )

    with tab2:
        if selected_document and selected_document != 'All':

            # Fast lookup of which collection the doc belongs to
            if selected_collection == 'All':
                selection_collection_placeholder = document_to_collection.get(selected_document)
            else:
                selection_collection_placeholder = selected_collection

            st.write(f"### Viewing: {selected_document}")

            pdf_path = f'json/{selection_collection_placeholder}/{selected_document}.pdf'

            try:
                with open(pdf_path, "rb") as pdf_file:
                    base64_pdf = base64.b64encode(pdf_file.read()).decode("utf-8")

                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="500" type="application/pdf"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)

            except FileNotFoundError:
                st.error(f"🚫 PDF file not found: `{pdf_path}`")

        else:
            st.warning("Please select a document to view.")

    with tab1:

        # Initialize session state for conversation history
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        # User input for the query
        user_input = st.chat_input("Ask me anything about the indexed documents!")

        if user_input:
            # Add the user input to the conversation history
            st.session_state.messages.append({"role": "user", "content": user_input})

            query_params = {
                "query_embeddings": cf.generate_query_embeddings(user_input),
                "pdf_name": None if selected_document == 'All' else selected_document,
                "document_collection_name": None if selected_collection == 'All' else selected_collection,
                "diversity": diversity,
                "top_n": top_n
            }



            # Retrieve top results based on the query embeddings
            results = cf.query_vector_dict(vector_dict, query_params)

            # Construct the prompt for the LLM
            prompt = f"""
                        Query: " {user_input} "
    
                        Top results:
                            {results['documents']}
                            
                        Please mention the Metadata below:
                            {results['metadata']}
                        
                        If the context does not provide enough information, reply by saying : 
                        'Please note that the current sources available to RAG are limited to indexed PDFs, so there may not be specific information related to your query. Apologies'  
                    """

            try:
                # Make the request to OpenAI to get the response
                reply = OpenAI().chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "developer", "content": "You are a helpful assistant"},
                        {"role": "user", "content": prompt}
                    ]
                )

                # Display the response
                assistant_reply = reply.choices[0].message.content
                st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

                st.write(results)

            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": f"Error generating response: {e}"})
        # Display chat history
        for message in st.session_state.messages:
            if message['role'] == "user":
                st.chat_message("user").markdown(message['content'])
            else:
                st.chat_message("assistant").markdown(message['content'])

"""

Developed by DSE Lab, Michigan State University
"""