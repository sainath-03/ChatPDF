import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Define correct passkey
correct_passkey = "sainath123"  # Replace with your actual passkey

# Function to authenticate the user using passkey
def authenticate_user():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("Welcome to Chat PDF with Gemini💁")

        passkey = st.text_input("🚀 Enter the passkey to proceed:", type="password")
        
        if st.button("Authenticate"):
            if passkey == correct_passkey:
                st.session_state.authenticated = True
                st.success("Authentication successful! Welcome aboard! 🎉")
                
            else:
                st.error("Authentication failed. Please enter a valid passkey. 🛑")
    return st.session_state.authenticated

# Function to get text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = (
        "system: You are an outstanding AI tutor known for your expertise and clarity. Your role is to assist students by providing clear, concise, and comprehensive answers to their assignment questions using the provided content from the uploaded document. Use your extensive knowledge only if necessary.\n"
        "user: Context: {context}\n\n"
        "Question: {question}\n\n"
        "assistant: \n"
        "Your answers should be:\n"
        "- Well-structured\n"
        "- Very detailed\n"
        "- Comprehensive for the topic\n"
        "- contain short and easy code snippets if needed in context to the topic of the document"
        "- Include relevant examples\n"
        "- Easy to understand for everyone\n"
        "- Free of any repetitive content and unnecessary length\n"
        "- Appropriate for an undergrad engineering AI and Data Science student\n"
        "Begin the answer by writing the question and then the answer"
        "Provide a clear, concise and comprehensive answer based on the PDF context"
    )

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    st.write("Reply: ", response["output_text"])

def main():
    if authenticate_user():
        st.set_page_config("Chat PDF")
        st.header("Chat with PDF using Gemini💁")

        user_question = st.text_input("Ask a Question from the PDF Files")

        if user_question:
            user_input(user_question)

        with st.sidebar:
            st.title("Menu:")
            pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
            if st.button("Submit & Process"):
                if pdf_docs:
                    with st.spinner("Processing..."):
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Done")
                else:
                    st.error("Please upload some PDF files.")

if __name__ == "__main__":
    main()
