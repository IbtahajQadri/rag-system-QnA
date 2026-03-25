import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

st.set_page_config(
    page_title="Automotive Safety Q&A",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Euro NCAP Safety Standards Q&A")
st.caption("Knowledge base: Euro NCAP Assessment Protocol – Safety Assist 2026")


@st.cache_resource
def load_retriever():
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever(search_type="similarity_score_threshold",search_kwargs={"k": 3, "score_threshold": 0.4})

@st.cache_resource
def load_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

retriever = load_retriever()
llm = load_llm()


prompt_template = PromptTemplate(
    template="""
You are an automotive safety expert assistant.
Use ONLY the following context to answer the question.
If the answer is not contained in the context, say:
"I could not find this information in the provided documents."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

def format_docs(docs):
    return "\n\n".join([
        f"[Page {doc.metadata['page']}]\n{doc.page_content}"
        for doc in docs
    ])

chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt_template
    | llm
    | StrOutputParser()
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

query = st.chat_input("Ask a question about automotive safety standards...")

if query:
    # Show user question in chat
    with st.chat_message("user"):
        st.write(query)

    # Save user question to history
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):

            # Run the chain
            answer = chain.invoke(query)

            # Get source pages
            source_docs = retriever.invoke(query)

            if not source_docs:
                st.write("I could not find relevant information in the provided documents.")
                st.stop()

            # Display answer
            st.write(answer)

            if "I could not find this information" not in answer:

                top_source = source_docs[0]
                # Display sources in expandable section
                with st.expander("📄 Most Relevant Source"):
                    
                    st.write(f"**{os.path.basename(top_source.metadata['source'])}")
                    st.write(f"**Extract:**")
                    st.write(top_source.page_content[:600] + "...")

    # Save answer to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })