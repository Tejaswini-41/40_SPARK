import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants and configurations
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Chatbot response templates
bot_template = """
<div class="bot-message">
    <p>{{MSG}}</p>
</div>
"""

css = """
<style>
    .bot-message {
        background-color: #d3eaf7;  /* Light blue */
        padding: 10px;
        border-radius: 5px;
        color: #000;  /* Text color */
        margin: 10px 0;
        font-size: 16px;
    }
</style>
"""

# Function to handle user input
def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please click the 'Process' button to initialize the conversation.")
        return

    # Get the chatbot response
    response = st.session_state.conversation({'question': user_question})
    
    # Store chat history
    st.session_state.chat_history.append({"user": user_question, "bot": response['answer']})

    # Display the chat history
    for message in st.session_state.chat_history:
        st.write(bot_template.replace("{{MSG}}", message['bot']), unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Code Search", page_icon='icon.jpg')

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'repositories' not in st.session_state:
        st.session_state.repositories = []

    st.write(css, unsafe_allow_html=True)

    st.header("Chat with Code", anchor="center")

    # User input for questions
    user_question = st.text_input("Which Code Snippet do you want to search?")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Enter Repository Details")

        # Inputs for repository name and project name
        if 'project_names' not in st.session_state:
            st.session_state.project_names = []

        if st.button("Add Repository"):
            # Add a new repository and project pair
            st.session_state.repositories.append({"repo": "", "project": ""})

        for index, repo_info in enumerate(st.session_state.repositories):
            col1, col2 = st.columns(2)

            with col1:
                repo_name = st.text_input(f"Repository Name {index + 1}", value=repo_info["repo"], key=f"repo_{index}")
            with col2:
                project_name = st.text_input(f"Project Name {index + 1}", value=repo_info["project"], key=f"project_{index}")

            # Update the repository info
            if repo_name and project_name:
                st.session_state.repositories[index] = {"repo": repo_name, "project": project_name}

        # Dropdown to select an active repository
        selected_repo_index = st.selectbox("Select Repository", range(len(st.session_state.repositories)), format_func=lambda x: f"{st.session_state.repositories[x]['repo']} - {st.session_state.repositories[x]['project']}")

        # Button to trigger processing
        if st.button("Process"):
            selected_repo = st.session_state.repositories[selected_repo_index]
            with st.spinner("Processing..."):
                # Replace this with your actual code fetching logic
                raw_text = get_code_from_repo(selected_repo["repo"], selected_repo["project"])

                # Split the raw text into chunks
                chunks = text_to_chunks(raw_text)

                # Create a vector store using the code chunks
                vectorstore = get_vectorstore(chunks)

                # Create conversation chain and store it in session state
                st.session_state.conversation = get_conversation_chain(vectorstore)

# Define your helper functions here
def get_code_from_repo(repo_name, project_name):
    # Implement your logic to fetch code from the repo
    return "Sample code from the repository."

def text_to_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separator="\n",
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

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

if __name__ == '_main_':
    main()