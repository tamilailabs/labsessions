import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
)

# =============================
# LANGCHAIN IMPORTS (STABLE)
# =============================
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# =============================
# CONFIGURATION
# =============================

PDF_PATH = "C:/Users/Prasanna/books/input/attention.pdf"   # 🔴 CHANGE THIS
INDEX_DIR = "faiss_index"

EMBED_MODEL = "BAAI/bge-base-en"
LLM_MODEL = "google/flan-t5-small"   # CPU friendly

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# =============================
# LOAD PDF
# =============================

print("Loading PDF...")
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

# =============================
# SPLIT DOCUMENT
# =============================

print("Splitting document...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)
docs = splitter.split_documents(documents)

# =============================
# LOAD EMBEDDINGS
# =============================

print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# =============================
# BUILD / LOAD FAISS
# =============================

if os.path.exists(INDEX_DIR):
    print("Loading FAISS index...")
    vectorstore = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(INDEX_DIR)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4}
)

# =============================
# LOAD LLM (CPU SAFE)
# =============================

print("Loading LLM (CPU)...")

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)

pipe = pipeline(
    task="text2text-generation",  # REQUIRED for T5
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=300,
    temperature=0.2
)

llm = HuggingFacePipeline(pipeline=pipe)

# =============================
# PROMPT TEMPLATE
# =============================

prompt = ChatPromptTemplate.from_template("""
You are a precise research assistant.

Use ONLY the provided context.
If answer not found, say "Not found in document."

Context:
{context}

Question:
{question}

Answer:
""")

parser = StrOutputParser()
chain = prompt | llm | parser

# =============================
# HELPER FUNCTION
# =============================

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# =============================
# INTERACTIVE LOOP
# =============================

print("\nRAG system ready. Type 'exit' to quit.\n")

while True:
    query = input("Question: ")

    if query.lower() == "exit":
        break

    # Retrieve relevant chunks
    retrieved_docs = retriever.invoke(query)
    context = format_docs(retrieved_docs)

    # Generate answer
    answer = chain.invoke({
        "context": context,
        "question": query
    })

    print("\nAnswer:\n")
    print(answer)

    print("\nSources:")
    for doc in retrieved_docs:
        page = doc.metadata.get("page", None)
        if page is not None:
            print(f"- Page {page + 1}")

    print("\n" + "="*60 + "\n")
