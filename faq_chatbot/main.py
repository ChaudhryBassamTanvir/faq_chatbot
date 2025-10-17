import os
import json
import nltk
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# -------------------------------
# 1️⃣  Setup
# -------------------------------
nltk.download("punkt", quiet=True)
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found in environment. Please set it before running.")

# -------------------------------
# 2️⃣  Load and prepare data
# -------------------------------
faq_file = "faqs.json"
if not os.path.exists(faq_file):
    raise FileNotFoundError("❌ 'faqs.json' file not found in the project directory.")

with open(faq_file, "r", encoding="utf-8") as f:
    faq_data = json.load(f)

# Convert to LangChain Documents
docs = [
    Document(page_content=f"Q: {faq['question']}\nA: {faq['answer']}")
    for faq in faq_data
]

# Split text (useful for long answers)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = splitter.split_documents(docs)

# -------------------------------
# 3️⃣  Create embeddings (Free Local Model)
# -------------------------------
print("🔍 Generating local embeddings using HuggingFace...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create or load Chroma database
db = Chroma.from_documents(split_docs, embeddings)

# -------------------------------
# 4️⃣  Initialize Gemini model (LLM)
# -------------------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

# -------------------------------
# 5️⃣  Interactive Chat Loop
# -------------------------------
print("\n🤖 FAQ Chatbot — Powered by Gemini & HuggingFace Embeddings")
print("Type 'exit' anytime to quit.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "exit":
        print("👋 Goodbye!")
        break

    if not user_input:
        print("⚠️ Please type a question or 'exit' to quit.\n")
        continue

    # Retrieve top matching FAQ(s)
    results = db.similarity_search(user_input, k=2)

    # Combine retrieved context
    context = "\n\n".join([doc.page_content for doc in results])

    # Generate a conversational response
    prompt = f"""You are a helpful assistant for the Mobile Finder System.
Use the following FAQs to answer the question naturally and clearly:

{context}

User: {user_input}
Answer:"""

    try:
        response = llm.invoke(prompt)
        print(f"Bot: {response.content}\n")
    except Exception as e:
        print(f"❌ Error generating response: {e}\n")
