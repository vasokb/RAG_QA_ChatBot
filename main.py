import torch
import streamlit as st
import time
import argparse
import requests
import tempfile

from utils import load_document, split_doc, create_vectorDB, load_llm, prompt_formatter

torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Available device: {device}")


def rag_chain(doc_path, query, reader_model, chunk_size=512, chunk_overlap=50, top_k=5, dtype=torch.bfloat16, device=device):
    # Load document
    doc = load_document(doc_path)
    print(f"The document consists of {len(doc)} pages")
    
    # Split document into chunks
    chunks = split_doc(doc, chunk_size, chunk_overlap)
    print(f"The document is split into {len(chunks)} chunks.")

    # Create embeddings
    print("Creation of embeddings and vector Database starts...")
    vectorDB = create_vectorDB(chunks, chunk_size, device)
    print("Vector DB is now created!")

    # Load the LLM wrapped in a pipeline
    READER_LLM, tokenizer = load_llm(reader_model, device, dtype)
    
    # Retrieve relevant docs and add them to the prompt template
    retrieved_docs = vectorDB.similarity_search(query=query, k=top_k)
    retrieved_docs_text = [doc.page_content for doc in retrieved_docs]  
    context = f"\n The {top_k} most relevant documents:\n"
    context += "\n\n".join([f"Chunk {str(i)}:\n" + doc for i, doc in enumerate(retrieved_docs_text)])

    augmented_prompt = prompt_formatter(query=query, tokenizer=tokenizer,
                            context_items=context)

    answer = READER_LLM(augmented_prompt)[0]["generated_text"]

    return answer, retrieved_docs_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RAG Application')
    parser.add_argument('--pdf', type=str, help='Path to PDF document.')
    args = parser.parse_args()

    if args.pdf is None:
        # Fetch master thesis pdf from url as default document
        url = "https://liu.diva-portal.org/smash/get/diva2:1573635/FULLTEXT01.pdf"
        response = requests.get(url)
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(response.content)
            args.pdf = temp_file.name

    st.title("Hi! I am an AI-powered chatbot. Let's chat! 👾")

    user_query = st.text_input("Enter your question:", "")
    if st.button("Submit"):
        try:
            start_time = time.time()
            response, relevant_docs = rag_chain(doc_path=args.pdf, query=user_query,
                                      reader_model='mistralai/Mistral-7B-Instruct-v0.1')
            end_time = time.time()
            execution_time = end_time - start_time
            st.write(f"{response} \n\n Execution time: {round(execution_time,2)} seconds")
        except Exception as e:
            st.write(f"An error occurred: {e}")