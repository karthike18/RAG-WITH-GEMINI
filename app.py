import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Directly enter your Google API key
def new_func():
    return "AIzaSyClN0GEDyp1XMcbQPbunADzBSuIqnHbfOE"

api_key = new_func()  # Replace with your actual API key

# Configure the Generative AI client with the API key
genai.configure(api_key="AIzaSyClN0GEDyp1XMcbQPbunADzBSuIqnHbfOE")

# Function to read all PDF files and return text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks  # list of strings

# Function to get embeddings for each chunk and store them in a vector store
def get_vector_store(chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", google_api_key="AIzaSyClN0GEDyp1XMcbQPbunADzBSuIqnHbfOE")  # Pass API key here
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.error(f"Error during vector store creation: {e}")
        st.stop()

# Function to create a conversational chain
def get_conversational_chain():
    try:
        prompt_template = """
        Answer the question as detailed as possible from the provided context. Make sure to provide all the details. If the answer is not in
        the provided context, just say, "Answer is not available in the context." Do not provide the wrong answer.\n\n
        Context:\n {context}?\n
        Question: \n{question}\n
        Answer:
        """
        
        model = ChatGoogleGenerativeAI(model="gemini-pro",
                                       google_api_key="AIzaSyClN0GEDyp1XMcbQPbunADzBSuIqnHbfOE",  # Pass API key here
                                       temperature=0.3)
        prompt = PromptTemplate(template=prompt_template,
                                input_variables=["context", "question"])
        chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Error during conversational chain creation: {e}")
        st.stop()

# Function to clear chat history
def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload some PDFs and ask me a question."}]

# Function to process user input and generate a response
def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", google_api_key="AIzaSyClN0GEDyp1XMcbQPbunADzBSuIqnHbfOE")  # Pass API key here

        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response
    except Exception as e:
        st.error(f"Error during user input processing: {e}")
        return None

# Main function to run the Streamlit app
def main():
    st.set_page_config(
        page_title="Gemini PDF Chatbot",
        page_icon="🤖"
    )

    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("EXPLORER:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing complete. You can now ask questions.")
            else:
                st.error("Please upload at least one PDF file.")

    # Main content area for displaying chat messages
    st.title("Chat with PDF files using RAG 🤖")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Placeholder for chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Upload some PDFs and ask me a question."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input and processing user questions
    prompt = st.chat_input()
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                if response:
                    full_response = response['output_text']
                    st.write(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    st.error("No response generated. Please try again.")

if __name__ == "__main__":
    main()
