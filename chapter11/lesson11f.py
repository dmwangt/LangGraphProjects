from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

# Initialize the embeddings model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Define sample documents as Document objects
documents = [
    Document(page_content="LangChain is a framework for building context-aware reasoning applications."),
    Document(page_content="FAISS is a library for efficient similarity search and clustering of dense vectors."),
    Document(page_content="The quick brown fox jumps over the lazy dog.")
]

# Index documents in Chroma vector store
db = Chroma.from_documents(documents, embedding=embeddings)
print("Documents indexed in Chroma successfully.")

# Define a retriever to fetch relevant documents
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# Define a prompt template for the LLM
prompt = ChatPromptTemplate.from_template(
"""
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""
)

# Initialize the chat model (GPT-4 variant or mini model for demonstration)
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai", temperature=0)


# Set up the RAG chain pipeline
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# Use the RAG pipeline to answer a user question
question = "What was the fox doing ?"
print(f"Question: {question}")
print(f"\n\n\ncontext: {retriever.get_relevant_documents(question)[0].page_content}\n\n-- end of context --\n\n")
# Stream the answer from the RAG chain
for chunk in rag_chain.stream(question):
    print(chunk, end="", flush=True)
