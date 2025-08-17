import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# --- Load vector store ---
embeddings = OllamaEmbeddings(model="all-minilm")
vectordb = Chroma(persist_directory="db", embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# --- Prompt template ---
prompt_template = """
You are a PriceLabs product specialist.
Always follow this structure:
1. Brief greeting (friendly + professional)
2. Explanation of the issue
3. Step-by-step resolution
4. Reference to supporting materials (if available)
5. Concise closing offering further assistance

Use the provided context to craft the response.
If the answer is not in the context, say "I don’t have that information."

Context:
{context}

Customer Question:
{question}

Answer in the style of the PriceLabs sample email.
"""
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# --- LLM ---
llm = Ollama(model="llama3.1", temperature=0)

# --- QA Chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

# --- Streamlit UI ---
st.set_page_config(page_title="PriceLabs Support Agent", page_icon="💬", layout="wide")
st.title("💬 PriceLabs Support Agent")

st.markdown("Ask any PriceLabs-style customer support question and get an answer following official guidelines.")

user_question = st.text_area("✏️ Enter Customer Email / Query", height=150)

if st.button("Generate Response"):
    if user_question.strip():
        with st.spinner("Generating response..."):
            response = qa_chain.run(user_question)
        st.subheader("📨 AI Response")
        st.write(response)
    else:
        st.warning("Please enter a question.")
