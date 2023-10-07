# Author: Daniel y Jose

from fastapi import FastAPI, File, UploadFile, Form, Request
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
import io

load_dotenv()

app = FastAPI()

class ChatRequest(BaseModel):
    chunks: List[str]
    question: str

# Enable CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You might want to restrict this to specific origins in production.
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload-pdf/")
async def upload_pdf(pdf: UploadFile = File(...)):
    try:
        print(pdf.filename)
        
        
        # Read the PDF file as bytes
        pdf_bytes = await pdf.read()
            
        # Create a BytesIO object to mimic a file-like object
        pdf_stream = io.BytesIO(pdf_bytes)
            
        # Use PdfReader to read the PDF from the BytesIO stream
        pdf_reader = PdfReader(pdf_stream)
            
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        chunks = text_splitter.split_text(text)
        return {"chunks": chunks}
    except Exception as e:
        logger.error(f"Error processing PDF upload: {str(e)}")
        return {"error": "An error occurred"}

@app.post("/answer-question/")
async def answer_question(request: Request, chat_request: ChatRequest):
    try:
        # Extract chunks and question from the request payload
        chunks = chat_request.chunks
        question = chat_request.question

        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        docs = knowledge_base.similarity_search(question)

        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=question)
            print(cb)

        # Return the response in JSON format
        return {"response": response}
    except Exception as e:
        # Handle any exceptions that may occur
        return {"error": str(e)}