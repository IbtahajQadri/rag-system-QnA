from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# Load the saved FAISS index
print("Loading vector store...")
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
print("Vector store loaded.")

# Build the retriever 
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# prompt template 
prompt_template = PromptTemplate(
    template="""
You are an automotive safety expert assistant.
Use ONLY the following context to answer the question.
If the answer is not contained in the context, say:
"I could not find this information in the provided documents."
Always mention which page the information came from.

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

# Helper to format retrieved chunks 
def format_docs(docs):
    return "\n\n".join([
        f"[Page {doc.metadata['page']}]\n{doc.page_content}"
        for doc in docs
    ])

# ── 5. Build the LLM 
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Build the LCEL (LangChain Expression Language) chain 
chain = (
    { "context": retriever | format_docs, "question": RunnablePassthrough() } | prompt_template | llm | StrOutputParser()
)

# ── 7. Question
query = "What are the criteria for blind spot monitoring?"

print(f"\nQuestion: {query}")
print("\nAnswer:")
answer = chain.invoke(query)
print(answer)

# ── 8. Show source pages
print("\nSources:")
source_docs = retriever.invoke(query)
for doc in source_docs:
    print(f"  - Page {doc.metadata['page']} | {doc.page_content[:100]}...")