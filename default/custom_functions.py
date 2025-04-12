import base64
import random
import numpy as np
import streamlit as st
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np
from openai import OpenAI
import os
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter



# Function to load and process JSON data from files
def load_json_files(json_files):
    ids = []
    documents = []
    metadata = []
    embeddings = []

    for json_file in json_files:
        # st.write(f"Loading JSON data from: {json_file}")
        try:
            # Open and read the JSON file
            with open(json_file, 'r') as file:
                json_data = json.load(file)  # Parse the JSON content
            # st.write(f"Successfully loaded JSON data from {json_file}")
        except Exception as e:
            st.write(f"Failed to load JSON data from {json_file}: {e}")
            continue  # Skip this file if reading fails

        # Process the data from the JSON
        try:
            # st.write("Processing JSON data.")
            ids.extend(json_data["ids"])  # Append data to the existing list
            documents.extend(json_data["documents"])
            metadata.extend(json_data["metadata"])
            embeddings.extend(json_data["embeddings"])
            # st.write(f"Processed data with {len(json_data['ids'])} rows successfully.")
        except KeyError as e:
            st.write(f"Error: Key {e} not found in the JSON data.")
        except Exception as e:
            st.write(f"Error processing the JSON data: {e}")

    # Convert embeddings to a NumPy array
    embeddings_array = np.array(embeddings)

    # Combine all data into a dictionary
    vector_dict = {
        "ids": ids,
        "documents": documents,
        "metadata": metadata,
        "embeddings": embeddings_array
    }

    return vector_dict


# Function to generate embeddings for a query using OpenAI API
def generate_query_embeddings(query_text):
    query_embeddings = OpenAI().embeddings.create(
        input=query_text,
        model="text-embedding-3-small"  # Specify the embedding model
    ).data[0].embedding

    query_embeddings = np.array(query_embeddings).reshape(1, -1)
    return query_embeddings

def query_vector_dict(vector_dict, query_params):

    new_dict = {
        'metadata' : [],
        'ids' : [],
        'embeddings' : [],
        'documents' : []
    }

    pdf_key_1 = query_params['pdf_name'] is not None
    collection_key_1 = query_params['document_collection_name'] is not None

    #filter knowledge base basis user parameters
    for i in range(len(vector_dict['ids'])):
        metadata = vector_dict['metadata'][i]

        pdf_key_2 = not pdf_key_1 or metadata['pdf_name'] == query_params['pdf_name']
        collection_key_2 = not collection_key_1 or metadata['document_collection_name'] == query_params['document_collection_name']

        if pdf_key_2 and collection_key_2 :
            new_dict['metadata'].append(vector_dict['metadata'][i])
            new_dict['ids'].append(vector_dict['ids'][i])
            new_dict['embeddings'].append(vector_dict['embeddings'][i])
            new_dict['documents'].append(vector_dict['documents'][i])

    #calculate similarity score and MMR(maximal marginal relevance) to fetch top_n documents
    l = query_params['diversity'] #lambda

    if query_params['top_n'] <= len(new_dict['ids']):
        top_n = query_params['top_n']
    else:
        top_n = len(new_dict['ids'])

    similarities = cosine_similarity(query_params['query_embeddings'], new_dict['embeddings'])
    mmr = l * similarities - (1 - l) * similarities.max(axis=1, keepdims=True)

    indices = np.argsort(mmr)[:,::-1][:,:top_n]

    final_dict = {
        'metadata' : [],
        'ids' : [],
        #'embeddings' : [],
        'documents' : []
    }

    #create final doct basis relevant and diverse indices
    for i in indices[0]:
        final_dict['metadata'].append(new_dict['metadata'][i])
        final_dict['ids'].append(new_dict['ids'][i])
        #final_dict['embeddings'].append(new_dict['embeddings'][i])
        final_dict['documents'].append(new_dict['documents'][i])

    return final_dict

def get_dict_list(vector_dict):
    result_dict = {}
    doc_list = []

    for metadata in vector_dict['metadata']:
        collection_name = metadata['document_collection_name']
        pdf_name = metadata['pdf_name']

        if collection_name not in result_dict:
            result_dict[collection_name] = []

        if pdf_name not in result_dict[collection_name]:
            result_dict[collection_name].append(pdf_name)

        if pdf_name not in doc_list:
            doc_list.append(pdf_name)

    return result_dict, doc_list


def load_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append(Document(page_content=content, metadata={"source": filename}))
    return documents


def split_text(documents, chunk_size, chunk_overlap, add_start_index):
    # Check for invalid overlap value
    if chunk_overlap > chunk_size:
        st.warning("Chunk Overlap cannot be greater than Chunk Size. Please adjust the values and try again.")
        return []

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=add_start_index,
        )
        chunks = text_splitter.split_documents(documents)
        st.write(f"Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks
    except ValueError as e:
        st.error(f"An error occurred while splitting the documents: {e}")
        return []