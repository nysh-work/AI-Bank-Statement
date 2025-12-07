'''
AI-Powered Bank Statement Auditor Tool
Audit bank statements using AI with Langchain and Google Gemini Pro

Features:
- Extract transaction data from bank statement PDFs
- Analyze transactions for anomalies and fraud indicators
- Answer natural language audit queries
- Generate audit findings and reports

Setup:
1. Obtain Google API key from https://makersuite.google.com/
2. Set GOOGLE_API_KEY in .env file
3. Note: VPN may be required in some regions

'''

import streamlit as st  # import stremlit
from  PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter # text splitter
import os


from langchain_google_genai import GoogleGenerativeAIEmbeddings # google gemini
import google.generativeai as genai # google gemini
# from langchain.vectorstores import FAISS # vector store use FAISS # oold version
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI # google gemini
from langchain.chains.question_answering import load_qa_chain # question answering chart
# from langchain.chains import RetrievalQAChain # retrieval question answering chart
from langchain.prompts import PromptTemplate # prompt template
# import langchain
from dotenv import load_dotenv # load environment variable


#load environment variable
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) # set key to environment variable

# Set up the model
generation_config = {
  "temperature": 0.55,#0.9,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}


safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]


#get pdf text
def getPDFText(pdf_docs):
    '''
    get pdf text from pdf docs
    '''
    text="" 
    for pdf in pdf_docs: #loop through pdf docs
        pdf_reader= PdfReader(pdf) #read pdf file
        for page in pdf_reader.pages: # loop through pdf pages 
            text+= page.extract_text() # extract text from page and add to text variable
    return  text # return text variable


#get text chunks (split text into chunks)
def getTextChunks(text, chunk_size=10000, chunk_overlap=1000):
    #inital Text splitter function 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                   chunk_overlap=chunk_overlap) # split text into chunks
    chunks = text_splitter.split_text(text) # split text into chunks
    return chunks # return chunks


#get vector store
def getVectorStore(text_chunks ,model="models/embedding-001"):
    embeddings = GoogleGenerativeAIEmbeddings(model = model) # use google gemini for embedding
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings) # use FAISS for vector store with google gemini embedding
    vector_store.save_local("faiss_index") # save vector store
    return vector_store # return vector store


#get conversational chain
def getConversationalChain(temperature=0.3, modelName="gemini-pro"):
    #promptTemplate
    promptTemplate = """
    You are an experienced financial auditor analyzing bank statements. Answer the audit question based on the provided bank statement data.

    Provide detailed, accurate responses including:
    - Specific transaction details (dates, amounts, parties, descriptions)
    - Any anomalies or unusual patterns detected
    - Relevant compliance observations
    - Supporting evidence from the statements

    If the answer is not available in the provided context, clearly state "The requested information is not available in the provided bank statements."
    Do not make assumptions or provide information not supported by the documents.

    Context (Bank Statement Data):\n {context}\n
    Audit Question: \n{question}\n

    Audit Response:
    """

    model=ChatGoogleGenerativeAI(model=modelName,
                                 temperature=temperature) # use google gemini model for conversational chain

    prompt = PromptTemplate(template = promptTemplate,
                            input_variables = ["context", "question"]) # use prompt template for conversational chain

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt) # load question answering chart

    return chain


# user input encode into query vector
def user_input(user_question, model= "models/embedding-001"):
    embeddings = GoogleGenerativeAIEmbeddings(model = model)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) # load vector store
    docs = new_db.similarity_search(user_question)  # search for similar text in vector store

    chain = getConversationalChain() # get conversational chain

    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)  # get response from conversational chain 
    
    print(response)
    st.write("Reply: ", response["output_text"])


# main function
def main():
    st.set_page_config("Bank Statement Auditor Tool") # set page config
    st.header("🔍 AI-Powered Bank Statement Audit Analysis Tool") #set header

    st.markdown("""
    ### Auditor Assistant
    This tool helps auditors analyze bank statements efficiently by:
    - Extracting and verifying transaction data
    - Detecting anomalies and unusual patterns
    - Identifying potential fraud indicators
    - Checking compliance with regulations
    - Generating comprehensive audit reports
    """)

    user_question = st.text_input("Ask an audit question (e.g., 'Show all transactions over $10,000', 'Are there any duplicate payments?', 'Identify unusual transactions')") # get user question

    if user_question:
        user_input(user_question) # get user input


    # sidebar
    with st.sidebar:
        st.title("📁 Upload Bank Statements") # set sidebar title
        st.markdown("Upload bank statement PDF files for audit analysis")
        pdfDocs = st.file_uploader("Upload Bank Statement PDF Files",
                                    accept_multiple_files=True,
                                    help="Upload one or more bank statement PDF files for comprehensive analysis") # upload pdf files
        # submit and process button
        if st.button("🔄 Process Statements"):
            with st.spinner("Extracting and analyzing bank statements..."):
                raw_text = getPDFText(pdfDocs) #
                text_chunks = getTextChunks(raw_text)
                getVectorStore(text_chunks)
                st.success("✅ Bank statements processed successfully! You can now ask audit questions.")

        st.markdown("---")
        st.markdown("### 📋 Sample Audit Queries")
        st.markdown("""
        - What is the total transaction amount?
        - Show all transactions over $5,000
        - Are there any duplicate payments?
        - Identify unusual or suspicious transactions
        - What is the account balance trend?
        - List all cash withdrawals
        - Find transactions with specific vendors
        """)


if __name__ == "__main__":
    main() # run main function