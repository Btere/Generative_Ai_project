import requests
import yfinance as yf
from bs4 import BeautifulSoup
from typing import List, Dict
import nltk
nltk.download('stopwords')  # For removing stopwords
nltk.download('punkt')  # For tokenization
nltk.download('punkt_tab')  # Specific resource causing the error
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from chromadb import Client
from langchain.llms import OpenAI



class PersonalizedPorfolioManager:
    def __init__(self, api_key: str, chroma_persist_dir: str = "./chroma_db"):
        """We initialize the PortfolioManager with ChromaDB for vector storage.

        Args:
            Api_key: API key for accessing financial APIs or embedding models.
            chroma_persist_dir: Directory to persist ChromaDB data.
        """
        self.api_key = api_key

        # Initialize ChromaDB client
        self.chroma_client = Client(
            Settings(
                persist_directory=chroma_persist_dir  # Adjust path as necessary
            )
        )

        # Initialize the collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="financial_data"
        )

    def fetch_yahoo_finance_dataset(self, symbols: List[str]) -> Dict:
        """Since we have different dataset, we fetchfinancial data for given stock symbols from yahoo finance.

        Args:
                List of stock symbols to fetch data for.

        Returns:
                Dictionary containing financial data.
        """
        data: Dict = {}
        for i, symbol in enumerate(symbols):
            try:
                price = 100 + i
                volume = 1000 * i
                data[symbol] = {"price": price, "volume": volume}
            except Exception as e:
                print(f"Error fetching data for symbol {symbol}: {e}")
        
        return data

    def scrape_blog_newsletter(self, url: str) -> str:
        """another dataset source,scraping and extract text content from a blog or newsletter.

        Args:
            url: URL of the blog or newsletter.

        Returns:
                Cleaned and preprocessed text content.
        """
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad responses
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text()  
            cleaned_text = self.data_cleaning(text)  # Preprocess the raw text
            return cleaned_text
        except requests.exceptions.RequestException as e:
            print(f"Error fetching URL {url}: {e}")
            return ""

    def data_cleaning(self, text: str) -> str:
        """We clean and preprocess text content.

        Args:
                text: Raw text content.

        Returns:
                Cleaned and preprocessed text.
        """
        # Download necessary NLTK resources
        nltk.download("stopwords")
        nltk.download("punkt")

        stop_words = set(stopwords.words("english"))
        tokens = word_tokenize(text)

        # Filter tokens to remove stopwords and non-alphanumeric tokens
        cleaned_tokens = [word for word in tokens if word.isalnum() and word.lower() not in stop_words]
        return " ".join(cleaned_tokens)

    def split_text(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
        """we splitted text into smaller chunks for better storage and retrieval.

        Args:
                text: Raw text to split.
                chunk_size: Maximum size of each chunk.
                chunk_overlap: Overlap between chunks to retain context.

        Returns:
                List of text chunks.
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_text(text)

    def store_text_vectors_to_chromadb(self, text: str, metadata: Dict):
        """we store text and metadata as vectors in ChromaDB or azure cosmodb.

        Args:
            text: Text content to vectorize and store.
            metadata: Metadata to associate with the text.
        """
        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[metadata.get("id", "default_id")]
        )

    def query_chromadb(self, query: str, n_results: int = 5) -> List[Dict]:
        """Query ChromaDB for similar text vectors.

        Args:
                query: Query text for similarity search.
                n_results: Number of results to retrieve.

        Returns:
                List of results from the database.
        """
        return self.collection.query(
            query_texts=[query],
            n_results=n_results
        )

    def generate_response(self, query: str, context: List[Dict]) -> str:
        """we generate a response using an LLM based on retrieved context.

        Args:
            query: User's input question.
            context: Retrieved chunks from ChromaDB.

        Returns:
                Generated response from the LLM.
        """
        # Concatenate retrieved context into a single string
        context_text = "\n\n".join([doc["document"] for doc in context])
        # create a prompt
        prompt = f"""You are a financial expert. Answer the following question based on the context provided."""

        model = OpenAI(temperature=0.7, api_key=self.api_key)
        return model(prompt)

    def ingest_data(self, url: str, metadata: Dict):
        """Ingest data from a blog or newsletter, preprocess, split, and store in ChromaDB.

        Args:
            url: URL of the blog or newsletter.
            metadata: Metadata for the document.
        """
        text = self.scrape_blog_newsletter(url)
        chunks = self.split_text(text)

        for idx, chunk in enumerate(chunks):
            chunk_metadata: Dict = {**metadata, "chunk_id": idx}
            self.store_text_vectors_to_chromadb(text=chunk, metadata=chunk_metadata)

def main():
    """Entry point to run the app"""
    api_key = "your_openai_api_key"
    portfolio_ = PersonalizedPorfolioManager(api_key)

    # Fetch Yahoo Finance dataset
    symbols = ["AAPL", "MSFT", "GOOGL"]
    finance_data = portfolio_.fetch_yahoo_finance_dataset(symbols)
    print("Finance Data:", finance_data)

    # Ingesting dataset from a blog
    blog_url = "https://example.com/financial-blog"
    portfolio_.ingest_data(blog_url, metadata={"source": "example_blog"})

    # Query ChromaDB/
    query = "what is the best stock to purchase in 2025, and what are the risks involve?"
    results = portfolio_.query_chromadb(query)
    print("Query Results:", results)

    # Generate a response based on query results
    response = portfolio_.generate_response(query, results)
    print("Generated Response:", response)


if __name__ == "__main__":
    main()