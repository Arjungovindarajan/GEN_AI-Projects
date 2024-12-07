from fastapi import FastAPI, HTTPException
import uvicorn
from bs4 import BeautifulSoup
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer, pipeline
import warnings
warnings.filterwarnings("ignore")


app = FastAPI()

# Load GPT-2 tokenizer and SentenceTransformer model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load OPT-125m model for text generation
generator = pipeline('text-generation', model="facebook/opt-125m")

# Global variables for storing website content, index, and embeddings
content_chunks = []
index = None
embeddings = None


# Function to fetch and scrape the website content
def fetch_website_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None


# Function to extract key information from different tags
def extract_key_information(soup):
    tags_to_extract = ['p', 'h1', 'h2', 'h3', 'li', 'td']  # Add more tags as necessary
    content = ""
    for tag in tags_to_extract:
        elements = soup.find_all(tag)
        content += "\n".join([element.get_text() for element in elements if element.get_text()])
    return content


# Function to create a FAISS index for the website content
def create_faiss_index(content_chunks):
    embeddings = embedder.encode(content_chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, embeddings


# Function to retrieve relevant content chunks based on user query
def retrieve_relevant_chunks(query, content_chunks, index, embeddings, top_k=2):
    query_embedding = embedder.encode([query])
    _, retrieved_indices = index.search(query_embedding, top_k)
    relevant_chunks = [content_chunks[idx] for idx in retrieved_indices[0]]
    return relevant_chunks


# Function to truncate content based on token limits
def truncate_content(content, max_tokens=700):
    tokens = tokenizer.encode(content)
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        truncated_content = tokenizer.decode(truncated_tokens)
        return truncated_content
    return content


# Function to generate a response using the OPT model
def chat_with_opt_model(prompt, max_length=100):
    result = generator(prompt, max_length=max_length, num_return_sequences=1)
    return result[0]['generated_text']

# Root route for the homepage ("/")
@app.get("/")
def read_root():
    print("Welcome to the FastAPI Chatbot API")
    return {"message": "Welcome to the FastAPI Chatbot API"}

# FastAPI route to initialize chatbot with website URL
@app.post("/initialize")
def initialize_chatbot(url: str):
    global content_chunks, index, embeddings

    # Fetch the website content
    soup = fetch_website_content(url)
    if not soup:
        raise HTTPException(status_code=400, detail="Error fetching the website content.")

    # Extract and process the website content
    website_content = extract_key_information(soup)
    content_chunks = website_content.split('\n\n')  # Split by paragraphs or double newlines

    # Create a FAISS index for the content
    index, embeddings = create_faiss_index(content_chunks)

    return {"message": "Chatbot initialized with website content."}


# FastAPI route to ask questions to the chatbot
@app.post("/ask")
def ask_chatbot(question: str):
    global content_chunks, index, embeddings

    if not content_chunks or not index:
        raise HTTPException(status_code=400, detail="Chatbot has not been initialized. Please initialize first.")

    # Retrieve relevant content from the website based on the query
    relevant_chunks = retrieve_relevant_chunks(question, content_chunks, index, embeddings, top_k=2)
    relevant_content = " ".join(relevant_chunks)

    # Truncate the relevant content to fit within token limits
    relevant_content = truncate_content(relevant_content, max_tokens=700)

    # Prepare the prompt
    prompt = f"Using the following information from the website:\n{relevant_content}\nUser: {question}"
    
    # Truncate the prompt to fit within token limits
    prompt = truncate_content(prompt, max_tokens=900)

    # Generate the chatbot's response
    response = chat_with_opt_model(prompt, max_length=100)

    return {"response": response}


# To run the FastAPI app with Uvicorn
if __name__ == "__main__":
    uvicorn.run("chatbot_app:app", host="0.0.0.0", port=8000, reload=True)
