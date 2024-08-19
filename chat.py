from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import json
import os
import pandas as pd
import string  # Added import
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

app = FastAPI()


pdf_loader = PyPDFLoader("Copy_of_About_Detail_Content.pdf")
pages = pdf_loader.load_and_split()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
context = "\n\n".join(str(p.page_content) for p in pages)
texts = text_splitter.split_text(context)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="AIzaSyD1UBGse7RYMujdj8aild7Gyp4GteBULbA")
vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": 5})
model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key="AIzaSyD1UBGse7RYMujdj8aild7Gyp4GteBULbA",
                             temperature=0.2, convert_system_message_to_human=True)
qa_chain = RetrievalQA.from_chain_type(
    model,
    retriever=vector_index,
    return_source_documents=True
)

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    detected_word: Optional[str] = None
    website_url: Optional[str] = None

@app.post("/ask_question", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    try:
        result = qa_chain({"query": request.question})
        answer = result["result"]
        
        # Detect word and get website URL
        detected_word = detect_word(answer)
        website_url = get_website_url(detected_word)
        
        return {"answer": answer, "detected_word": detected_word, "website_url": website_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def detect_word(sentence):
    words = [
        "CEMENT", "AGRI PERISHABLES", "COAL", "AGRICULTURE BIOMASS",
        "BUILDING AND FINISHING", "COTTON AND YARN", "ENERGY AND PETROLEUM",
        "FERTILIZERS", "GRAINS AND PULSES", "STEEL", "CHEMICALS", "SUGER"
    ]
    sentence = sentence.upper()
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    s = sentence.split()

    for word in words:
        if word in s:
            return word
    return None

def get_website_url(detected_word):
    urls = {
        "CEMENT": "https://zarea.pk/product-category/cement/",
        "AGRI PERISHABLES": "https://zarea.pk/product-category/agri-perishables-fruits-rates-today/",
        "COAL": "https://zarea.pk/product-category/coal-price-today-in-pakistan/",
        "AGRICULTURE BIOMASS": "https://zarea.pk/product-category/agriculture-biomass/",
        "BUILDING AND FINISHING": "https://zarea.pk/product-category/building-and-finishing/",
        "COTTON AND YARN": "https://zarea.pk/product-category/cotton-yarn/",
        "ENERGY AND PETROLEUM": "https://zarea.pk/product-category/energy-petroleum/",
        "FERTILIZERS": "https://zarea.pk/product-category/fertilizers/",
        "GRAINS AND PULSES": "https://zarea.pk/product-category/grains-pulses/",
        "STEEL": "https://zarea.pk/product-category/steel/",
        "SUGER": "https://zarea.pk/product-category/sugar/",
        "CHEMICALS": "https://zarea.pk/product-category/chemicals/"
    }
    return urls.get(detected_word, "https://zarea.pk/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
