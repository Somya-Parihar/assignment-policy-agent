# Multi-Role Insurance AI Agent

A robust AI agent built with **LangGraph**, **LangChain**, and **Gemini** that seamlessly switches between three distinct roles:
1.  **Support Agent (RAG):** Answers policy questions using a PDF knowledge base.
2.  **Lead Qualifier (State Machine):** Collects specific user details (Age, Location, Income) with validation logic.
3.  **Agentic Tool Caller:** Generates an insurance quote via a mock API once qualification is complete.

## Features
* **Intelligent Routing:** Automatically detects if the user is asking a question or looking to buy.
* **State Management:** Uses a Finite State Machine (FSM) to ensure all lead data is collected naturally, one by one.
* **Validation:** strict checks on inputs (e.g., Age must be 0–110).
* **Persistent Memory:** Saves conversation history to `checkpoints.db` (SQLite) so you can close and resume chats.
* **Efficient RAG:** Embeds PDFs locally using FAISS and HuggingFace, persisting the index to disk to avoid re-processing.

---

## Installation

### 1. Clone the Repository
Clone this repository to your local machine:
```bash
git clone <your-repo-url>
cd insurance-agent
```
### 2. Install Dependencies
Make sure you have Python 3.9+ installed. Then install the required packages:
```Bash
pip install -r requirements.txt
```
## Configuration
### 1. Set up API Keys
You need a Google Gemini API key to run the LLM.

1. Find the file named sample.env in the root directory.

2. Open it and paste your API Key:

    ```Code snippet
    GEMINI_API_KEY=your_actual_api_key_here
    ```
3. Rename the file from sample.env to .env.

### 2. Add Policy Document
Place your insurance policy PDF in the root folder and name it policy.pdf. (If no PDF is found, the agent will initialize with an empty knowledge base, but it will not crash.)

## Usage
1. To start the agent in your terminal:
```Bash
python main.py
```
2. to start the agent in api mode
```Bash
python api.py
```
this will be a web page on `localhost:8000`


Interaction Guide
* Ask Questions: "Does this policy cover theft?" (Triggers RAG)

* Get a Quote: "I want to buy insurance." (Triggers Lead Qualification)

* Quit: Type quit or exit to close the application.
