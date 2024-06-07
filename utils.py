from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import pipeline

import streamlit as st

st.cache_data
def load_doc(doc_path):
    """Load the PDF document and return the document object."""
    loader = PyPDFLoader(doc_path, extract_images=False)
    doc = loader.load()
    # doc[2].page_content
    return doc


st.cache_data
def split_doc(doc, chunk_size=512, chunk_overlap=50):
    """Split the document into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""],
        add_start_index=True,  # includes chunk's start index in metadata
        strip_whitespace=True  # strips whitespace from the start and end of every document
    )

    chunks = text_splitter.split_documents(doc)
    return chunks


st.cache_resource
def create_vectorDB(chunks, chunk_size, device, embedding_model='thenlper/gte-small', dist_strategy=DistanceStrategy.COSINE):
    """Embed the chunks and store them in a vector database."""
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model,
                                       model_kwargs={'device': device},
                                       # need to normalize the embeddings for cosine similarity
                                       encode_kwargs={
                                           "normalize_embeddings": True}
                                       )

    emb_model_max_seq_len = SentenceTransformer(embedding_model).max_seq_length
    if emb_model_max_seq_len < chunk_size:
        raise ValueError(
            "The chunk_size is bigger than the max_seq_length of the embedding model! Choose a different embeddings model or change chunk_size.")

    vectorDB = FAISS.from_documents(
        chunks, embeddings, distance_strategy=dist_strategy)
    return vectorDB


def context_retrieval(vectorDB, query, top_k=5):
    """Retrieve the most relevant documents based on the query."""
    retrieved_docs = vectorDB.similarity_search(query=query, k=top_k)
    retrieved_docs_text = [doc.page_content for doc in retrieved_docs]
    context = f"\n The {top_k} most relevant documents:\n"
    context += "\n\n".join([f"Chunk {str(i)}:\n" +
                           doc for i, doc in enumerate(retrieved_docs_text)])
    return retrieved_docs_text, context


st.cache_resource(ttl='1d')
def load_model(reader_model, device, dtype):
    """Load the LLM and tokenizer."""
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True)  # load in 4bit, compute in float16
    tokenizer = AutoTokenizer.from_pretrained(reader_model)
    model = AutoModelForCausalLM.from_pretrained(reader_model,
                                                 device_map=device,
                                                 torch_dtype=dtype,
                                                 quantization_config=quantization_config,
                                                 low_cpu_mem_usage=True  # if we do not have gpu available this can be False
                                                 )

    READER_LLM = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        # Samples from the probability distribution of the predictions, instead of choosing token with the highest probability.
        do_sample=True,
        temperature=0.1,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=512,
    )

    return READER_LLM, tokenizer


def prompt_formatter(query, tokenizer, context_items):
    """Augment the user query with document-based context."""

    context_augmented_prompt = f"""Based on the information contained in the context items, please answer the query.
    Give yourself room to think by extracting relevant passages from the context before answering the query.
    Don't return the thinking, only return the answer.
    Make sure your answers are as explanatory as possible.
    If the answer cannot be deduced from the context, say that you are not completely sure before giving the answer.

    \nNow use the following context items to answer the user query:
    {context_items}

    \nThis is the user query you need to answer.
    User query: {query}
    Answer:"""

    # Create prompt template for instruction-tuned model
    dialogue_template = [
        {"role": "user",
         "content": context_augmented_prompt}
    ]

    # Apply the chat template
    prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                           tokenize=False,
                                           add_generation_prompt=True)
    return prompt
