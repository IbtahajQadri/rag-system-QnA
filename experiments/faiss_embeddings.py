from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
loader = PyPDFLoader("data/euro-ncap-assessment-protocol-sa-collision-avoidance-v1041.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)
print(f"Chunks ready: {len(chunks)}")


# create embeddings and store in FAISS
print("\nCreating embeddings and storing in FAISS...")
print("(This calls OpenAI API — may take 10-20 seconds)")
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save the vectorstore locally
vectorstore.save_local("faiss_index")
print("FAISS index created and saved locally as 'faiss_index'")

# Test run of semantic search
print("\nRunning test search...")
query = "What are the criteria for blind spot monitoring?"
results = vectorstore.similarity_search(query, k=3)

print(f"\nTop 3 results for query: '{query}'")
for i, doc in enumerate(results):
    print(f"\n--- Chunk {i+1} (page {doc.metadata['page']}) ---")
    print(doc.page_content[:500])  # Print first 500 chars of the chunk