# Wanna know more about my Master's thesis? Why don't you ask my personal AI assistant? :robot:

This project is aimed at creating an interactive Question Answering (QA) system using the Retrieval Augmented Generation (RAG) technique. This ChatBot combines the power of information retrieval with natural language generation to provide accurate and informative responses to user queries. Leveraging state-of-the-art LLMs, the application retrieves relevant context from the input document and generates concise answers to queries. 

As an AI enthusiast, I developed this ChatBot application as a personal project to explore the capabilities of RAG and further develop my skills in Generative AI. I applied it on my Master's thesis document as a way to inject a bit of fun and excitement into something I already cherished. Plus, who knows how much I'll remember about it in a few years?

The application is developed using the following frameworks:

**Hugging Face**: Hugging Face's Transformers library is used for accessing pre-trained models, as well as for tokenization and model inference.

**LangChain**: LangChain is the framework used for all the building blocks of RAG, such as text splitting and vector database creation.

**Streamlit**: Streamlit is used to create a user interface, enabling users to interact with the chatbot through a web browser.

## Status

This project is currently under development.

#### TODO

- [ ] Setup an evaluation pipeline
- [ ] Experiment with more sophisticated chunking methods, embedding models, etc.
- [ ] Enable users to upload any document through the UI

## Usage

First clone the repository using Git.

Then install all the dependencies using the following command:

```bash
pip install -r requirements.txt
````

To start the chatbot run the following command:

```bash
streamlit run main.py
````

After running the command, the chatbot will open in a browser. Then you can start interacting with the chatbot and ask it questions.

## Demo

Add a visual demo here.