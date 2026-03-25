import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

print("Loading PDFs from data/ folder...")
all_chunks = []
data_folder = "data"

for filename in os.listdir(data_folder):
    if filename.endswith(".pdf"):
        filepath = os.path.join(data_folder, filename)
        print(f"  Loading: {filename}")
        loader = PyPDFLoader(filepath)
        pages = loader.load()
        print(f"  → {len(pages)} pages loaded")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(pages)
        print(f"  → {len(chunks)} chunks created")
        all_chunks.extend(chunks)


# create embeddings and store in FAISS
print("\nCreating embeddings and storing in FAISS...")
print("(This calls OpenAI API — may take 10-20 seconds)")
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(all_chunks, embeddings)

# Save the vectorstore locally
vectorstore.save_local("faiss_index")
print("FAISS index created and saved locally as 'faiss_index'")
