# LlamaIndex Chatbot with Google Gemini

A Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, LlamaIndex, and Google Gemini. This application allows you to chat with your documents using advanced AI models.

## Features

- Interactive chat interface with multiple chat sessions
- RAG-powered responses using document embeddings
- Streaming responses for real-time interaction
- Persistent chat history
- Support for PDF and other document formats

## Prerequisites

- Python 3.8 or higher
- Google Gemini API key

## Installation

1. Clone or download this repository.

2. Create and activate a virtual environment:
   ```
   python3 -m venv .venv
   ```

3. Activate the virtual environment:
   for macOS / Linux
   ```
   source .venv/bin/activate  
   ```
   for Windows/Linux
   ```
   .venv\Scripts\activate
   ```

5. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

6. Create a `.env` file in the root directory and add your Google Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

7. Create a folder named `data/`

## Usage

1. Place your documents (PDFs, text files, etc.) in the `data/` folder.

2. Run the Streamlit application:
   ```bash
   streamlit run chat.py
   ```

3. Open your browser to the provided URL.

4. Start chatting with your documents!

## Project Structure

- `chat.py`: Main Streamlit application
- `requirements.txt`: Python dependencies
- `data/`: Directory for your documents
- `chat_history.json`: Stores chat session history (auto-generated)

## Technologies Used

- [Streamlit](https://streamlit.io/) - Web app framework
- [LlamaIndex](https://www.llamaindex.ai/) - Data framework for LLM applications
- [Google Gemini](https://gemini.google.com/) - AI model for generation and embeddings
- [python-dotenv](https://pypi.org/project/python-dotenv/) - Environment variable management

## License

This project is open-source. Feel free to modify and distribute.
