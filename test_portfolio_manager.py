import unittest
from unittest.mock import patch, Mock, MagicMock
import pytest
import requests
from bs4 import BeautifulSoup
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prototype import PersonalizedPorfolioManager


class TestPersonalizedPortfolioManager(unittest.TestCase):
    """Test suite for PersonalizedPorfolioManager class"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.api_key = "test_api_key"
        self.test_chroma_dir = "./test_chroma_db"
        
        # Mock ChromaDB client to avoid actual database operations
        with patch('prototype.Client') as mock_client:
            self.mock_collection = Mock()
            mock_client.return_value.get_or_create_collection.return_value = self.mock_collection
            self.portfolio_manager = PersonalizedPorfolioManager(
                api_key=self.api_key,
                chroma_persist_dir=self.test_chroma_dir
            )
    
    def tearDown(self):
        """Clean up after each test method."""
        # Clean up any test files if they exist
        import shutil
        if os.path.exists(self.test_chroma_dir):
            shutil.rmtree(self.test_chroma_dir)
    
    def test_init(self):
        """Test initialization of PersonalizedPorfolioManager"""
        self.assertEqual(self.portfolio_manager.api_key, self.api_key)
        self.assertIsNotNone(self.portfolio_manager.chroma_client)
        self.assertIsNotNone(self.portfolio_manager.collection)
    
    def test_fetch_yahoo_finance_dataset(self):
        """Test fetching financial data from Yahoo Finance"""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        result = self.portfolio_manager.fetch_yahoo_finance_dataset(symbols)
        
        # Verify the structure and content
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), len(symbols))
        
        for i, symbol in enumerate(symbols):
            self.assertIn(symbol, result)
            self.assertEqual(result[symbol]["price"], 100 + i)
            self.assertEqual(result[symbol]["volume"], 1000 * i)
    
    def test_fetch_yahoo_finance_dataset_empty_list(self):
        """Test fetching data with empty symbol list"""
        result = self.portfolio_manager.fetch_yahoo_finance_dataset([])
        self.assertEqual(result, {})
    
    @patch('prototype.requests.get')
    def test_scrape_blog_newsletter_success(self, mock_get):
        """Test successful blog scraping"""
        # Mock successful HTTP response
        mock_response = Mock()
        mock_response.content = b"<html><body><p>Test financial content</p></body></html>"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Mock the data_cleaning method to return predictable output
        with patch.object(self.portfolio_manager, 'data_cleaning', return_value="Test financial content"):
            url = "https://example.com/blog"
            result = self.portfolio_manager.scrape_blog_newsletter(url)
            
            self.assertEqual(result, "Test financial content")
            mock_get.assert_called_once_with(url)
    
    @patch('prototype.requests.get')
    def test_scrape_blog_newsletter_request_exception(self, mock_get):
        """Test blog scraping with request exception"""
        mock_get.side_effect = requests.exceptions.RequestException("Connection error")
        
        url = "https://invalid-url.com"
        result = self.portfolio_manager.scrape_blog_newsletter(url)
        
        self.assertEqual(result, "")
    
    @patch('prototype.requests.get')
    def test_scrape_blog_newsletter_http_error(self, mock_get):
        """Test blog scraping with HTTP error"""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        mock_get.return_value = mock_response
        
        url = "https://example.com/not-found"
        result = self.portfolio_manager.scrape_blog_newsletter(url)
        
        self.assertEqual(result, "")
    
    @patch('prototype.nltk.download')
    @patch('prototype.word_tokenize')
    @patch('prototype.stopwords.words')
    def test_data_cleaning(self, mock_stopwords, mock_tokenize, mock_download):
        """Test text cleaning functionality"""
        # Mock NLTK dependencies
        mock_stopwords.return_value = ["the", "is", "and"]
        mock_tokenize.return_value = ["This", "is", "a", "test", "text", "123", "!"]
        
        text = "This is a test text 123!"
        result = self.portfolio_manager.data_cleaning(text)
        
        # Should return tokens that are alphanumeric and not stopwords
        expected_tokens = ["This", "a", "test", "text", "123"]
        self.assertEqual(result, " ".join(expected_tokens))
    
    def test_split_text(self):
        """Test text splitting functionality"""
        text = "A" * 1000  # 1000 character string
        chunks = self.portfolio_manager.split_text(text, chunk_size=200, chunk_overlap=50)
        
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 1)  # Should split into multiple chunks
        
        # Test with small text that doesn't need splitting
        small_text = "Short text"
        small_chunks = self.portfolio_manager.split_text(small_text)
        self.assertEqual(len(small_chunks), 1)
        self.assertEqual(small_chunks[0], small_text)
    
    def test_store_text_vectors_to_chromadb(self):
        """Test storing text vectors to ChromaDB"""
        text = "Test financial document"
        metadata = {"id": "test_doc_1", "source": "test"}
        
        self.portfolio_manager.store_text_vectors_to_chromadb(text, metadata)
        
        # Verify ChromaDB collection.add was called with correct parameters
        self.mock_collection.add.assert_called_once_with(
            documents=[text],
            metadatas=[metadata],
            ids=["test_doc_1"]
        )
    
    def test_store_text_vectors_default_id(self):
        """Test storing text vectors with default ID"""
        text = "Test document without ID"
        metadata = {"source": "test"}
        
        self.portfolio_manager.store_text_vectors_to_chromadb(text, metadata)
        
        # Should use default_id when no id in metadata
        self.mock_collection.add.assert_called_once_with(
            documents=[text],
            metadatas=[metadata],
            ids=["default_id"]
        )
    
    def test_query_chromadb(self):
        """Test querying ChromaDB"""
        query = "best stocks to buy"
        mock_results = {
            "documents": [["Stock analysis document"]],
            "metadatas": [[{"source": "blog"}]],
            "distances": [[0.1]]
        }
        self.mock_collection.query.return_value = mock_results
        
        result = self.portfolio_manager.query_chromadb(query, n_results=3)
        
        self.assertEqual(result, mock_results)
        self.mock_collection.query.assert_called_once_with(
            query_texts=[query],
            n_results=3
        )
    
    @patch('prototype.OpenAI')
    def test_generate_response(self, mock_openai):
        """Test response generation with OpenAI"""
        # Mock OpenAI model
        mock_model = Mock()
        mock_model.return_value = "Generated financial advice"
        mock_openai.return_value = mock_model
        
        query = "What stocks should I buy?"
        context = [
            {"document": "AAPL is performing well"},
            {"document": "Tech stocks are rising"}
        ]
        
        result = self.portfolio_manager.generate_response(query, context)
        
        self.assertEqual(result, "Generated financial advice")
        mock_openai.assert_called_once_with(temperature=0.7, api_key=self.api_key)
    
    @patch.object(PersonalizedPorfolioManager, 'scrape_blog_newsletter')
    @patch.object(PersonalizedPorfolioManager, 'split_text')
    @patch.object(PersonalizedPorfolioManager, 'store_text_vectors_to_chromadb')
    def test_ingest_data(self, mock_store, mock_split, mock_scrape):
        """Test data ingestion workflow"""
        # Mock the workflow
        mock_scrape.return_value = "Scraped financial content"
        mock_split.return_value = ["chunk1", "chunk2", "chunk3"]
        
        url = "https://example.com/blog"
        metadata = {"source": "financial_blog"}
        
        self.portfolio_manager.ingest_data(url, metadata)
        
        # Verify the workflow calls
        mock_scrape.assert_called_once_with(url)
        mock_split.assert_called_once_with("Scraped financial content")
        
        # Verify store was called for each chunk
        self.assertEqual(mock_store.call_count, 3)
        
        # Verify that each chunk was stored with correct metadata
        # Check first call
        first_call = mock_store.call_args_list[0]
        self.assertEqual(first_call[1]['text'], "chunk1")
        self.assertEqual(first_call[1]['metadata'], {**metadata, "chunk_id": 0})
        
        # Check second call
        second_call = mock_store.call_args_list[1] 
        self.assertEqual(second_call[1]['text'], "chunk2")
        self.assertEqual(second_call[1]['metadata'], {**metadata, "chunk_id": 1})
        
        # Check third call
        third_call = mock_store.call_args_list[2]
        self.assertEqual(third_call[1]['text'], "chunk3")
        self.assertEqual(third_call[1]['metadata'], {**metadata, "chunk_id": 2})


class TestIntegration(unittest.TestCase):
    """Integration tests for the portfolio manager"""
    
    @patch('prototype.Client')
    def test_full_workflow_integration(self, mock_client):
        """Test the complete workflow integration"""
        # Setup
        mock_collection = Mock()
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        
        manager = PersonalizedPorfolioManager("test_key")
        
        # Test finance data fetch
        symbols = ["AAPL", "MSFT"]
        finance_data = manager.fetch_yahoo_finance_dataset(symbols)
        self.assertEqual(len(finance_data), 2)
        
        # Test that manager is properly initialized
        self.assertIsNotNone(manager.chroma_client)
        self.assertIsNotNone(manager.collection)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)