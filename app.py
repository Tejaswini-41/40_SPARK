import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from azure.devops.connection import Connection
from msrest.authentication import BasicAuthentication
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")    

genai.configure(api_key=GOOGLE_API_KEY)

#PAT for Azure Devops 
personal_access_token = os.getenv("AZURE_DEVOPS_PAT")  
organization_url = 'https://dev.azure.com/shwetambhosale18'

# Authenticate using the personal access token
credentials = BasicAuthentication('', personal_access_token)
connection = Connection(base_url=organization_url, creds=credentials)

# Initialize the Git client
git_client = connection.clients.get_git_client()

def add_custom_css():
    st.markdown(
        """
        <style>
        .center-button {
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }
        .stButton button {
            width: 300px; /* Adjust this width as per your preference */
            font-size: 16px; /* Optional: Increase the font size for better readability */
            padding: 10px 20px; /* Optional: Add some padding for a larger button */
        }
        </style>
        """, 
        unsafe_allow_html=True
    )


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

def get_code_from_repo(repo_name, project_name, file_types=['.py', '.js','.txt','java','.md']):
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
        api_key=""
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain

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

    
def show_sidebar():
    """Sidebar content that remains visible across pages"""
    with st.sidebar:
        st.subheader("Enter Repository Details")

        if 'repositories' not in st.session_state:
            st.session_state.repositories = []

        if st.button("Add Repository"):
            st.session_state.repositories.append({"repo": "", "project": ""})

        for index, repo_info in enumerate(st.session_state.repositories):
            col1, col2 = st.columns(2)
            with col1:
                repo_name = st.text_input(f"Repository Name {index + 1}", value=repo_info["repo"], key=f"repo_{index}")
            with col2:
                project_name = st.text_input(f"Project Name {index + 1}", value=repo_info["project"], key=f"project_{index}")

            if repo_name and project_name:
                st.session_state.repositories[index] = {"repo": repo_name, "project": project_name}

        selected_repo_index = st.selectbox(
            "Select Repository", range(len(st.session_state.repositories)),
            format_func=lambda x: f"{st.session_state.repositories[x]['repo']} - {st.session_state.repositories[x]['project']}"
        )

        # Button to process repository
        if st.button("Process"):
            selected_repo = st.session_state.repositories[selected_repo_index]
            if not selected_repo["repo"] or not selected_repo["project"]:
                st.error("Both repository name and project name must be provided.")
            else:
                with st.spinner("Processing..."):
                    raw_text = get_code_from_repo(selected_repo["repo"], selected_repo["project"])
                    if raw_text is None:
                        st.error(f"Repository '{selected_repo['repo']}' or project '{selected_repo['project']}' does not exist.")
                    else:
                        chunks = text_to_chunks(raw_text)
                        vectorstore = get_vectorstore(chunks)
                        st.session_state.conversation = get_conversation_chain(vectorstore)

        # Button to navigate to the Error Checking page
        if st.button("Error Checking"):
            st.session_state.page = "error_check"  # Navigate to error check page

        # Button to know company standards
        if st.button("Know Company Standards"):
            st.session_state.page = "standards"  # Navigate to company standards page

        if st.button("Personalized Code Assistance"):
            st.session_state.page = "personalized_assistance"

        # Button to show bot info
        if st.button("Bot Info"):
            st.session_state.page = "info"  # Navigate to info page

def show_main_page():
    """Main page content"""
    st.header("Chat with Code", anchor="center")

    # User input for questions
    user_question = st.text_input("Which Code Snippet do you want to search?")
    if user_question:
        handle_userinput(user_question)


def show_personalized_assistance_page():
    """Personalized Code Assistance page content"""
    st.title("Personalized Code Assistance")

    # File uploader for code file
    uploaded_file = st.file_uploader("Upload your code file", type=["py", "js", "java", "c", "cpp", "html", "css"])

    if uploaded_file is not None:
        # Read the content of the uploaded file
        code_content = uploaded_file.read().decode("utf-8")
        st.text_area("Your Code", value=code_content, height=300)

        if st.button("Ask about this code"):
            if code_content:
                with st.spinner("Getting assistance..."):
                    result = handle_userinput(f"Analyze this code:\n{code_content}")
                    st.write(result)  # Display the bot's response

def show_standards_page():
    """Company standards page content"""
    st.title("Company Standards")

    # Clear previous information before checking new standard
    if 'standard_info' in st.session_state:
        del st.session_state['standard_info']

    # Text input for company standard query
    company_standard = st.text_input("Which company standard would you like to know about?")

    st.markdown('<div class="center-button">', unsafe_allow_html=True)
    if st.button("Get Info"):
        if company_standard:
            with st.spinner("Retrieving information..."):
                # Perform info retrieval using the handle_userinput function
                result = handle_userinput(f"What are company standards for {company_standard}.")
                
                # Store the result in session state to manage state across reruns
                st.session_state['standard_info'] = result

    # Display the retrieved information
    if 'standard_info' in st.session_state:
        st.write(st.session_state['standard_info'])

    st.markdown('</div>', unsafe_allow_html=True)


def show_error_check_page():
    """Error Checking page content"""
    st.title("Error Checking")

    # Clear previous errors before checking new code
    if 'error_check_result' in st.session_state:
        del st.session_state['error_check_result']

    # Text area for error checking code
    code_to_check = st.text_area("Enter your code here to check for errors:")

    st.markdown('<div class="center-button">', unsafe_allow_html=True)
    if st.button("Check for Errors"):
        if code_to_check:
            with st.spinner("Checking for errors..."):
                # Perform error checking using the handle_userinput function
                result = handle_userinput(f"Check this code for errors:\n{code_to_check}")
                
                # Store the result in session state to manage state across reruns
                st.session_state['error_check_result'] = result

    # Display error check result if available
    if 'error_check_result' in st.session_state:
        st.write(st.session_state['error_check_result'])

        # Ask user for feedback
        st.markdown('<div class="center-button">', unsafe_allow_html=True)
        st.write("Do you like this response?")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Like"):
                st.session_state['feedback'] = "like"
                st.success("Thanks for your feedback!")
        with col2:
            if st.button("👎 Dislike"):
                st.session_state['feedback'] = "dislike"
                st.warning("We'll work on improving this.")

        st.markdown('</div>', unsafe_allow_html=True)

    # Button to go back to the main page
    st.markdown('<div class="center-button">', unsafe_allow_html=True)
    if st.button("Back to Main"):
        st.session_state.page = "main"  # Navigate back to the main page
    st.markdown('</div>', unsafe_allow_html=True)

def display_pdf(pdf_file_path):
    """Function to display the PDF content directly in the app"""
    pdf_display = f'<iframe src="data:application/pdf;base64,{pdf_to_base64(pdf_file_path)}" width="140%" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


def pdf_to_base64(pdf_file_path):
    """Convert the PDF file to base64 to embed in iframe"""
    import base64
    with open(pdf_file_path, "rb") as pdf_file:
        base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
    return base64_pdf

def show_info_page():
    """Info page content"""
    st.title("Bot Usage Information")
    display_pdf('bot_info.pdf')  # Display the PDF file here

def main():
    st.set_page_config(page_title="Code Search", page_icon='icon.jpg')

    # Add custom CSS for buttons
    add_custom_css()

    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "main"  # Default page to main

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'repositories' not in st.session_state:
        st.session_state.repositories = []
    

    st.write(css, unsafe_allow_html=True)

    # Sidebar content remains visible on both pages
    show_sidebar()

    # Navigate between pages
    if st.session_state.page == "main":
        show_main_page()
    elif st.session_state.page == "error_check":
        show_error_check_page()
    elif st.session_state.page == "standards":  
        show_standards_page()
    elif st.session_state.page == "personalized_assistance":
        show_personalized_assistance_page() 

    elif st.session_state.page == "info":
        show_info_page() 


if __name__ == '__main__':
    main()
