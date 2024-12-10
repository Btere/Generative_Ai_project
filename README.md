# Project Summary

This repository contains a Python implementation of a Personalized Portfolio Manager designed to optimize and personalize investment portfolios using AI-driven techniques. The focus is on developing a Market Information Agent that collects and cleans data from various sources to provide valuable insights to clients.

## Problem Statement
Financial institutions often struggle with managing and utilizing diverse financial data effectively. Challenges include data inconsistencies, incomplete datasets, and the need for personalized investment strategies tailored to individual client preferences and market conditions.

### Solution Overview
The primary use case is developing a Market Information Agent that, the Personalized Portfolio Manager addresses these challenges through the following features, gathers data from financial APIs and textual sources, cleans and preprocesses data to maintain accuracy and provides insights and recommendations to clients, enhancing decision-making in investment management through the following steps.

1. Data Collection: Fetches financial data from APIs (e.g., Yahoo Finance) using specified stock symbols.
2. Text Scraping and Cleaning: Extracts and preprocesses text from blogs or newsletters to derive insights.
3. Text Chunking and Storage: Splits text into manageable chunks for efficient storage and retrieval(RAG) using ChromaDB.
4. AI-Driven Response Generation: Utilizes OpenAI's language models (like GPT) to generate context-aware responses for client queries and collected data.


### Repository Contents
prototype.py: Python script showcasing the functionality of the Personalized Portfolio Manager.
README.md: Documentation providing an overview of the project, installation instructions, and usage guidelines.
requirement.txt: File for storing all installed libraries.

# Prerequisites

Before running the prototype, ensure the following are installs:
Python Environment:
   - Install Python 3.8 or later.
   - Install langchain framework

## Virtual environment
Virtual environment helps isolate the dependencies required for a project and avoids conflicts with global Python packages. Open your terminal and navigate to the directory where your prototype.py script is located, then type the command to create an environment.

```
python -m venv env
```
This will create a folder named env (or your chosen name) that contains a fresh Python environment.

Once the virtual environment is created, you need to activate it to start using the isolated environment.

#### For Windows

```
.\env\Scripts\activate
```

#### For Macs

```
source env/bin/activate
```

When activated, your terminal prompt will change to show the name of the virtual environment, e.g., (env).


### Install Required Libraries

Libraries:
   Install the required Python libraries using `pip3`:
   ```bash
   pip3 install requests beautifulsoup4 nltk langchain chromadb openai
   ```

### Create the requirements.txt File

After installing the necessary libraries, run the following command:
```
pip freeze > requirements.txt

```
This will create a requirements.txt file containing the exact versions of the libraries you just installed.

 Set an Environment Variables:
   - Get an API key for OpenAI.
   - Set it as an environment variable or pass it directly to the code.

### Set Up the API Key for OpenAI (Environment Variable)
To avoid hardcoding your OpenAI API key into your script, you can use an environment variable to securely store the API key.

### Set the environment variable:
For Windows:
Run the following command in the terminal (in the same terminal session where you're working):

```
bash
Copy code
set OPENAI_API_KEY=your_openai_api_key_here
```

For macOS/Linux:
Run the following command in the terminal:
```
bash
Copy code
export OPENAI_API_KEY="your_openai_api_key_here"
```

Replace your_openai_api_key_here with your actual API key from OpenAI.


### NLTK Resources:
   Run the following once to download NLTK stopwords and tokenizer resources:
   ```python
   import nltk
   nltk.download("stopwords")
   nltk.download("punkt")
   ```

### Setting Up ChromaDB

1. **Create a Persistent Directory**:
   - The code uses `./chroma_db` as the default directory. Ensure this directory exists, or change the path in the `PersonalizedPorfolioManager` constructor.

2. **Run ChromaDB Locally**:
   - ChromaDB runs as a local database, so no additional setup is needed. Ensure you have write permissions to the chosen directory.

### Running the Prototype

1. Clone or Copy the Code:
   Save the `PersonalizedPorfolioManager` class into a Python file, e.g., `prototype.py`.

2. Prepare Input Data:
   - For scraping: Use a valid blog/newsletter URL/endpoint
   - For querying: Prepare any financial question or topic of interest.

3. Run the Script:
    Run the Python script:
   ```bash
   python3 prototype.py
   ```

### Expected Output
If the prototype runs successfully, you should see:
1. **Ingestion Confirmation**:
   Logs or prints confirming that data from the blog was processed and stored in ChromaDB.

2. **Query Results**:
   Retrieved chunks or documents matching your query from ChromaDB.

   Example:
   ```json
   [
       {"document": "Tech stocks are on the rise due to AI advancements.", "metadata": {"source": "example_blog"}},
       {"document": "The S&P 500 shows steady growth over the last quarter.", "metadata": {"source": "newsletter"}}
   ]
   ```

#### Note

We can use bloomberg gpt api as the model to use, it is a  50- billion parameters llm model built from scratch for finance.
We can also use Azure cosmodb for the vector database for RAG.

### Troubleshooting
- Missing Dependencies: Ensure all libraries are installed.
- API Key Errors:
  - Verify the API key for OpenAI or your chosen LLM.
  - Check if the LLM service is up and reachable.
- Scraping Issues:
  - Ensure the blog/newsletter URL is valid and accessible.
  - If the blog structure changes, the scraper may need adjustments.
- ChromaDB Errors:
  - Ensure the `chromadb` library is properly configured and the `chroma_db` directory is writable.
