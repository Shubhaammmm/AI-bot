from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_community.vectorstores import Qdrant
from langchain_ollama import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv
load_dotenv()
# ---------- LangChain Setup ----------

# Embeddings and client
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

qdrant_client = QdrantClient(
    url="https://5061f8bd-e30a-45fc-9dde-44d131b1d5e2.us-east4-0.gcp.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.g2ks_AiOpHt7P-j0uDUOUUTkMkEC8gpwo-efqxEo9uc"
)

vectorstore = Qdrant(
    collection_name="maharashtra_begging_act",
    client=qdrant_client,
    embeddings=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Prompt template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a legal assistant specialized in Indian law. Use the context from the Maharashtra Prevention of Begging Act to answer the question concisely and accurately.

Context:
{context}

Question:
{question}

Answer:"""
)

llm = ChatGroq(
    model="llama3-8b-8192",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    max_tokens=512,  # increase if you need longer answers
    temperature=0.2
)

# llm = ChatOllama(model="llama3:latest")

llm_chain = LLMChain(llm=llm, prompt=prompt)
stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")

retrieval_chain = RetrievalQA(
    retriever=retriever,
    combine_documents_chain=stuff_chain,
    return_source_documents=True
)

# ---------- FastAPI Setup ----------

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "Maharashtra Begging Act QA API is running."}

@app.post("/ask")
def ask_question(payload: QuestionRequest):
    result = retrieval_chain({"query": payload.question})
    answer = result["result"]
    sources = [doc.page_content[:300] for doc in result["source_documents"]]

    return {
        "question": payload.question,
        "answer": answer,
        "sources": sources
    }
