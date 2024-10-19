from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from htmlTemp import css, bot_template, user_template
import google.generativeai as genai
from azure.devops.connection import Connection
from msrest.authentication import BasicAuthentication
from langchain_groq import ChatGroq
import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify


from dotenv import load_dotenv
load_dotenv()


#PAT
personal_access_token = os.getenv("AZURE_DEVOPS_PAT")  
organization_url = 'https://dev.azure.com/shwetambhosale18'

# Authenticate using the personal access token
credentials = BasicAuthentication('', personal_access_token)
connection = Connection(base_url=organization_url, creds=credentials)

# Initialize the Git client
git_client = connection.clients.get_git_client()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")    

genai.configure(api_key=GOOGLE_API_KEY)

app = Flask(_name_)

def get_code_from_repo(repo_name, project_name, file_types=['.py', '.js']):
    repo = git_client.get_repository(project=project_name, repository_id=repo_name)
    
    # Get items (files and directories) from the repository root
    items = git_client.get_items(project=project_name, repository_id=repo.id, recursion_level='Full')

    if not items:
        print("No items found in the repository.")
    
    code = ""
    for item in items:
        if item.is_folder:
            print(f"Skipping folder: {item.path}")
        else:
            print(f"Found file: {item.path}")  # Debug print to show found file paths
            
            if any(item.path.endswith(ext) for ext in file_types):  # Filter by file type
                print(f"Fetching content of: {item.path}")  # Debug print to show file being processed
                
                # Get the content of each item (handle as a generator)
                file_content_generator = git_client.get_blob_content(project=project_name, repository_id=repo.id, sha1=item.object_id)
                
                # Collect content from the generator and decode bytes
                file_content = ''.join([chunk.decode('utf-8') for chunk in file_content_generator])

                if file_content:
                    code += file_content  # Append the content to the code string
                else:
                    print(f"No content found for file: {item.path}")  # Debug print if no content is found
    
    return code

def text_to_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
    separator = "\n",
    length_function = len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

conversation = None
chat_history = []
app.secret_key = b'\xac\xaa\xd6\xa2\x83\xf7h\xf4\xc9\xd7Q/\xf4@\x12\xfb\xff*\xf8\xe6b{\xc5\xc0'

def get_conversation_chain(vectorstore):
    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key="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"
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain

def custom_response_handler(user_question):
    """Prepare a custom response based on user input."""
    if "code" in user_question.lower():
        # Prepare a prompt that asks only for code
        prompt = f"Retrieve code snippets related to: {user_question}. Please return the code only, without any explanations."
    elif "explain" in user_question.lower() or "optimize" in user_question.lower():
        # Prepare a prompt that asks for explanation or optimization
        prompt = f"Explain or optimize the following code: {user_question}."
    else:
        # Fallback if the input is unclear
        prompt = f"Provide a response to: {user_question}."

    # Call the LLM with the prepared prompt
    response = conversation({'question': prompt})  # Call the conversation chain with the custom prompt
    return response

@app.route('/create_project', methods=['POST'])
def create_project():
    """Endpoint to create a project (if needed for API functionality)."""
    data = request.get_json()
    repository_name = data.get('repository_name')
    project_name = data.get('project_name')
    
    return jsonify({"repository": repository_name, "project": project_name})

@app.route('/', methods=['GET'])
def index():
    """Render the main chat interface."""
    global chat_history
    return render_template('index.html', chat_history=chat_history)

@app.route('/ask', methods=['POST'])
def ask():
    """Handle user questions."""
    global conversation, chat_history
    user_question = request.form['user_question']

    if conversation is None:
        flash("Please process the repository first.", "warning")
        return redirect(url_for('index'))

    # Process user question using the custom response handler
    response = custom_response_handler(user_question)
    
    # Update chat history
    chat_history.append(response['chat_history'])  # Append the new message to chat history
    return redirect(url_for('index'))

@app.route('/process', methods=['POST'])
def process_repo():
    global conversation, chat_history
    project_name = request.form['project_name']
    repo_name = request.form['repo_name']

    if project_name and repo_name:
        raw_text = get_code_from_repo(repo_name, project_name)
        chunks = text_to_chunks(raw_text)
        vectorstore = get_vectorstore(chunks)
        conversation = get_conversation_chain(vectorstore)
        chat_history = []  # Clear previous chat history
        flash("Repository processed successfully!", "success")
    else:
        flash("Please provide both the project name and repository name.", "warning")

    return redirect(url_for('index'))

if __name__ == '_main_':
    app.run(debug=True)