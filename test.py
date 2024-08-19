import urllib
import warnings
from pathlib import Path as p
from pprint import pprint

import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA



warnings.filterwarnings("ignore")




from langchain_google_genai import ChatGoogleGenerativeAI
GOOGLE_API_KEY="AIzaSyD1UBGse7RYMujdj8aild7Gyp4GteBULbA"
model = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=GOOGLE_API_KEY,
                             temperature=0.2,convert_system_message_to_human=True)


pdf_loader = PyPDFLoader("Copy_of_About_Detail_Content.pdf")
pages = pdf_loader.load_and_split()

from langchain_google_genai import GoogleGenerativeAIEmbeddings

text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
context = "\n\n".join(str(p.page_content) for p in pages)
texts = text_splitter.split_text(context)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GOOGLE_API_KEY)
vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k":5})
qa_chain = RetrievalQA.from_chain_type(
    model,
    retriever=vector_index,
    return_source_documents=True

)
import json
import os
import pandas as pd
from IPython.display import Markdown

data_file = 'qa_data.json'
csv_file = 'qa_data.csv'

if os.path.exists(data_file):
    with open(data_file, 'r') as f:
        qa_dict = json.load(f)
else:
    qa_dict = {}

question = input("Ask a Question: ")

result = qa_chain({"query": question})

qa_dict[question] = result["result"]

result_text = result["result"]
Markdown(result_text)


with open(data_file, 'w') as f:
    json.dump(qa_dict, f)

df = pd.DataFrame(list(qa_dict.items()), columns=['Question', 'Answer'])
df.to_csv(csv_file, index=False)
print(result_text)
print()
import string

def isWordPresent(sentence, words):
    sentence = sentence.upper()
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    s = sentence.split()

    for word in words:
        if word in s:
            return word
    return None

if __name__ == "__main__":
    sentence = result_text
    words = [
        "CEMENT", "AGRI PERISHABLES", "COAL", "AGRICULTURE BIOMASS",
        "BUILDING AND FINISHING", "COTTON AND YARN", "ENERGY AND PETROLEUM",
        "FERTILIZERS", "GRAINS AND PULSES", "STEEL", "CHEMICALS" , "SUGER"
    ]

    detected_word = isWordPresent(sentence, words)

    if detected_word == "CEMENT":
        print("Kindly Visit Our Site->https://zarea.pk/product-category/cement/")
    elif detected_word == "AGRI PERISHABLES":
        print("Visit Our Site->https://zarea.pk/product-category/agri-perishables-fruits-rates-today/")
    elif detected_word == "COAL":
        print("Visit Our Site->https://zarea.pk/product-category/coal-price-today-in-pakistan/")
    elif detected_word == "AGRICULTURE BIOMASS":
        print("Visit Our Site->https://zarea.pk/product-category/agriculture-biomass/")
    elif detected_word == "BUILDING AND FINISHING":
        print("Visit Our Site->https://zarea.pk/product-category/building-and-finishing/")
    elif detected_word == "COTTON AND YARN":
        print("Visit Our Site->https://zarea.pk/product-category/cotton-yarn/")
    elif detected_word == "ENERGY AND PETROLEUM":
        print("Visit Our Site->https://zarea.pk/product-category/energy-petroleum/")
    elif detected_word == "FERTILIZERS":
        print("Visit Our Site->https://zarea.pk/product-category/fertilizers/")
    elif detected_word == "GRAINS AND PULSES":
        print("Visit Our Site->https://zarea.pk/product-category/grains-pulses/")
    elif detected_word == "STEEL":
        print("Visit Our Site->https://zarea.pk/product-category/steel/")
    elif detected_word == "SUGER":
        print("Visit Our Site->https://zarea.pk/product-category/sugar/")
    elif detected_word == "CHEMICALS":
        print("Visit Our Site->https://zarea.pk/product-category/chemicals/")
    else:
        print("Feel Free to visit our Website->https://zarea.pk/")

Markdown(result_text)