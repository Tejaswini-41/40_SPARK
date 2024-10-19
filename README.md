# Code Query Chatbot with Azure DevOps Integration

This project is a ** WEB*-based application that enables users to chat with code repositories stored in **Azure DevOps. The chatbot can retrieve and process code, answer queries related to it, and even check for errors. It leverages **Google Generative AI* for embedding generation and *FAISS* for similarity search, allowing users to interactively ask questions and retrieve code snippets or explanations.

## Features

- Fetches code from Azure DevOps repositories.
- Splits large codebases into smaller chunks for efficient processing.
- Embeds code snippets using *Google Generative AI* embeddings.
- Supports conversational query handling via the *ChatGroq* model.
- Error checking for code on user request.
- Retrieves company standards and guidelines with pre-defined options.
- Provides an interactive Streamlit-based interface for user interaction.

## Tech Stack

- *Streamlit* for the frontend interface.
- *Flask* for backend API integration.
- *Azure DevOps API* for fetching code repositories.
- *Google Generative AI* for embeddings and text generation.
- *FAISS* for similarity search in code snippets.
- *ChatGroq* for conversational AI.

## Installation

1. *Clone the repository:*
   ```bash
   git clone https://github.com/yourusername/repository-name.git
   cd repository-name
Create a virtual environment:

bash
Copy code
python3 -m venv env
source env/bin/activate  # For Windows use env\Scripts\activate
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Set up environment variables (see below).

Environment Variables
Create a .env file in the root directory with the following variables:

GOOGLE_API_KEY=your-google-api-key
GROQ_API_KEY=your-groq-api-key
AZURE_DEVOPS_PAT=your-azure-devops-personal-access-token
AZURE_DEVOPS_ORG_URL=your-organization-url
These values are necessary for integrating with Google Generative AI, ChatGroq, and Azure DevOps.

Usage
Run the Streamlit app:

streamlit run app.py
Process the Repository:

On the web interface, select a repository and a project from the dropdown.
Click the 'Process' button to fetch the code and initialize the conversation.
Ask a Question:

Type a question related to the repository code in the input box and submit it.
The bot will retrieve relevant code snippets or provide explanations based on your query.
Error Checking:

Click the Error Check button to verify code correctness in real-time.
Company Standards:

Use the Know Company Standards button to retrieve guidelines.
Features and Functionality
Code Retrieval from Azure DevOps:
Fetches and processes code from a given repository and file types (.py, .js, etc.).

Conversational Interface:
Users can ask questions about the code, and the system will provide relevant answers or code snippets based on vector search.

Error Checking:
Users can check for potential code errors by clicking a dedicated button that processes and validates the code.

Memory-Powered Conversations:
The bot uses a memory buffer to maintain the context of ongoing conversations.

Company Standards Integration:
An easy interface to ask about specific company standards related to coding practices.
