from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from pymongo import MongoClient
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from azure.devops.connection import Connection
from msrest.authentication import BasicAuthentication

app = FastAPI()
load_dotenv(dotenv_path=".env")

# CORS middleware setup
origins = [
    "http://localhost:5173",  
    "http://localhost:3000"    
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = MongoClient(os.getenv("MONGO_URI"))
db = client.get_database(os.getenv("DBDB_NAME"))
students_collection = db.get_collection("repos")

# Azure DevOps PAT and organization URL
personal_access_token = os.getenv("AZURE_DEVOPS_PAT")
organization_url = 'https://dev.azure.com/shwetambhosale18'

# Authenticate using the personal access token
credentials = BasicAuthentication('', personal_access_token)
connection = Connection(base_url=organization_url, creds=credentials)

# Initialize the Git client
git_client = connection.clients.get_git_client()

# Define data models
class Repository(BaseModel):
    project_name: str
    repo_name: str
    github_url: str

class AskRequest(BaseModel):
    user_question: str

# Function to fetch code from a repository
def get_code_from_repo(repo_name, project_name, file_types=['.py', '.js']):
    repo = git_client.get_repository(project=project_name, repository_id=repo_name)
    
    items = git_client.get_items(project=project_name, repository_id=repo.id, recursion_level='Full')
    
    if not items:
        print("No items found in the repository.")
    
    code = ""
    for item in items:
        if not item.is_folder and any(item.path.endswith(ext) for ext in file_types):
            file_content_generator = git_client.get_blob_content(project=project_name, repository_id=repo.id, sha1=item.object_id)
            file_content = ''.join([chunk.decode('utf-8') for chunk in file_content_generator])
            code += file_content if file_content else ''
    
    return code

# Function to split text into chunks
def text_to_chunks(raw_text):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separator="\n", length_function=len)
    chunks = text_splitter.split_text(raw_text)
    return chunks

# Function to create a vector store from chunks
def get_vectorstore(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

conversation = None
app.secret_key = os.urandom(24)

# Function to get the conversation chain
def get_conversation_chain(vectorstore):
    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=os.getenv("CHAT_GROQ_API_KEY")
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    return conversation_chain

# Endpoint to process repository information
@app.post("/process")
async def process_repository(repo: Repository):
    code = get_code_from_repo(repo.repo_name, repo.project_name)
    chunks = text_to_chunks(code)
    vectorstore = get_vectorstore(chunks)
    global conversation
    conversation = get_conversation_chain(vectorstore)
    return {"message": "Repository processed successfully."}

# Endpoint to handle user questions
@app.post("/ask")
async def ask_question(request: AskRequest):
    user_question = request.user_question
    
    if conversation is None:
        raise HTTPException(status_code=400, detail="No conversation chain available. Please process a repository first.")
    
    # Determine the prompt type based on the user's question
    if "code" in user_question.lower():
        prompt = f"Retrieve code snippets related to: {user_question}. Please return the code only, without any explanations."
    elif "explain" in user_question.lower() or "optimize" in user_question.lower():
        prompt = f"Explain or optimize the following code: {user_question}."
    else:
        prompt = f"Provide a response to: {user_question}."
    
    response = conversation({'question': prompt})
    
    # Return the final response to the user
    return {"answer": response.get('answer', 'No response generated.')}

# Root endpoint to check connection
@app.get("/")
async def read_root():
    return {"message": "Hello, Checking connection"}

