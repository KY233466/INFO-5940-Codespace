# RAG APP 
This application enables users to upload multiple documents in either PDF or TXT format. For each uploaded document, a dedicated RAG database is automatically created, allowing the system to retrieve information specific to that document.

During a conversation, users can selectively choose which documents to include in the retrieval process using on-screen toggles. The chatbot then generates responses grounded only in the selected documents. For transparency, each user message displays a “Document used:” label showing which documents were referenced for that question.

# Changes made to the provided configurations
Specified pandas version and added PyPDF2

# Getting Started

### Step 1: Open your forked repo Codespace
1. Click the green **Code** button and switch to the **Codespaces** tab.  
2. Select **Create Codespace**.
3. Wait a few minutes for the environment to finish setting up.

### Step 3: Verify your environment 
Once the Codespace is ready: 
1. If you are in `<your-file-name>.ipynb` in your codespace.
2. Install the Python 3.11.13 Kernel.  In the top-right corner, click **Select Kernel**.
    1. If **Install/Enable suggested extensions Python + Jupyter** appears, select it, and wait for the install to finish before moving on to the next step.
    2. Select **Python Environments** choose **Python 3.11.13 (first option)**.
3. Run the code block to check your setup. 
  
## Running a Streamlit App on Codespaces  
Follow these steps to launch and view your Streamlit app in GitHub Codespaces:
1. **Open the terminal** inside your Codespace.
2. Run the command:  
   ```bash
   API_KEY=<your_actual_API_KEY> streamlit run rag_app.py
   ```
3. After pressing **Enter**, a popup should appear in the bottom-right corner of Codespace editor.  
   - Click **“Open in Browser”** to view your app.  

   ⚠️ *If you miss the popup:*  
   - Press **Ctrl + C** in the terminal to stop the app.  
   - Rerun the command from step 2 — the popup should appear again.
4. A new browser tab will open, showing the interface of your Streamlit app.
5. **Make changes to your code** in the Codespace editor.  
   - Refresh the browser tab to see the updated version of your app.  