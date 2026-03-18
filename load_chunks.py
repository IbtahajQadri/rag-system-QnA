from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
loader = PyPDFLoader("data/euro-ncap-assessment-protocol-sa-collision-avoidance-v1041.pdf")
pages = loader.load()

print(f"Total pages loaded: {len(pages)}")
print(f"\nSample text from page 11:\n{pages[11].page_content[:500]}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(pages)

print(f"Total chunks created: {len(chunks)}")
print(f"\nSample chunk 1:\n{chunks[40].page_content}")
print(f"\nChunk metadata: {chunks[40].metadata}")