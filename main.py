import torch
import streamlit as st
import time
import argparse
import requests
import tempfile

from utils import load_doc, split_doc, create_vectorDB, load_model, prompt_formatter, context_retrieval

torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Available device: {device}")


class RAGChain:
    def __init__(self, doc_path, reader_model, chunk_size=512, chunk_overlap=50, top_k=5, dtype=torch.bfloat16, device=device):
        self.doc_path = doc_path
        self.reader_model = reader_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.dtype = dtype
        self.device = device

    def load_document(self):
        doc = load_doc(self.doc_path)
        print(f"The document consists of {len(doc)} pages")
        return doc

    def split_document(self, doc):
        chunks = split_doc(doc, self.chunk_size, self.chunk_overlap)
        print(f"The document is split into {len(chunks)} chunks.")
        return chunks

    def create_embeddings(self, chunks):
        print("Creation of embeddings and vector Database starts...")
        vectorDB = create_vectorDB(chunks, self.chunk_size, self.device)
        print("Vector DB is now created!")
        return vectorDB

    def retrieve_docs(self, vectorDB, query):
        retrieved_docs, context = context_retrieval(
            vectorDB, query, top_k=self.top_k)
        return retrieved_docs, context

    def load_llm(self):
        READER_LLM, tokenizer = load_model(
            self.reader_model, self.device, self.dtype)
        return READER_LLM, tokenizer

    def format_prompt(self, query, tokenizer, context):
        augmented_prompt = prompt_formatter(
            query=query, tokenizer=tokenizer, context_items=context)
        return augmented_prompt

    def generate_answer(self, augmented_prompt, READER_LLM):
        answer = READER_LLM(augmented_prompt)[0]["generated_text"]
        return answer

    def rag_chain(self, query):
        doc = self.load_document()
        chunks = self.split_document(doc)
        vectorDB = self.create_embeddings(chunks)
        retrieved_docs, context = self.retrieve_docs(vectorDB, query)
        READER_LLM, tokenizer = self.load_llm()
        augmented_prompt = self.format_prompt(query, tokenizer, context)
        answer = self.generate_answer(augmented_prompt, READER_LLM)
        return answer


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
            rag = RAGChain(args.pdf, 'mistralai/Mistral-7B-Instruct-v0.1')
            response = rag.rag_chain(user_query)
            end_time = time.time()
            execution_time = end_time - start_time
            st.write(
                f"{response} \n\n Execution time: {round(execution_time,2)} seconds")
        except Exception as e:
            st.write(f"An error occurred: {e}")
