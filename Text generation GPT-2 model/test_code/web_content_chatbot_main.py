import requests
from bs4 import BeautifulSoup
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer, pipeline
import warnings
warnings.filterwarnings("ignore")

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load a SentenceTransformer model for embedding the website content
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load OPT-125m model for text generation
generator = pipeline('text-generation', model="facebook/opt-125m")

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
    # Extracting text from various important tags
    tags_to_extract = ['p', 'h1', 'h2', 'h3', 'li', 'td']  # Add more tags as necessary
    content = ""
    
    for tag in tags_to_extract:
        elements = soup.find_all(tag)
        content += "\n".join([element.get_text() for element in elements if element.get_text()])
    
    return content

# Function to create a FAISS index for the website content
def create_faiss_index(content_chunks):
    # Compute embeddings for each chunk
    embeddings = embedder.encode(content_chunks)
    
    # Initialize FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    
    # Add the embeddings to the index
    index.add(np.array(embeddings))
    
    return index, embeddings

# Function to retrieve relevant content chunks based on user query
def retrieve_relevant_chunks(query, content_chunks, index, embeddings, top_k=2):
    query_embedding = embedder.encode([query])
    
    # Search the FAISS index for the most relevant chunks
    _, retrieved_indices = index.search(query_embedding, top_k)
    
    # Return the most relevant chunks
    relevant_chunks = [content_chunks[idx] for idx in retrieved_indices[0]]
    return relevant_chunks

# Function to truncate the content dynamically based on token limits
def truncate_content(content, max_tokens=700):
    tokens = tokenizer.encode(content)
    
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        truncated_content = tokenizer.decode(truncated_tokens)
        return truncated_content
    return content

# Function to generate a response using OPT model
def chat_with_opt_model(prompt, max_length=100):
    result = generator(prompt, max_length=max_length, num_return_sequences=1)
    return result[0]['generated_text']

# Main chatbot function
def chatbot(url = 'https://botpenguin.com/'):

    # Fetch the website data
    soup = fetch_website_content(url)
    if not soup:
        return
    
    # Extract and process key data
    website_content = extract_key_information(soup)
    
    # Split the website content into chunks (for retrieval)
    content_chunks = website_content.split('\n\n')  # Split by paragraphs or double newlines
    
    # Create a FAISS index for the content chunks
    index, embeddings = create_faiss_index(content_chunks)
    
    print(f"Website content fetched and processed. FAISS index created.\n")

    # User interaction loop
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Exiting the chatbot.")
            break
        
        # Retrieve relevant chunks using the FAISS index
        relevant_chunks = retrieve_relevant_chunks(user_input, content_chunks, index, embeddings, top_k=2)
        
        # Join the relevant chunks into a single string
        relevant_content = " ".join(relevant_chunks)
        
        # Truncate the relevant content to fit within the token limit
        relevant_content = truncate_content(relevant_content, max_tokens=700)  # Reserve space for user input
        
        # Prepare the prompt for the text generation model using the truncated relevant content
        prompt = f"Using the following information from the website:\n{relevant_content}"
        
        # Truncate the prompt if it exceeds the token limit
        prompt = truncate_content(prompt, max_tokens=900)  # Limit input tokens to 900, leaving room for the model to generate
        
        # Generate the response using the OPT model
        response = chat_with_opt_model(prompt, max_length=100)
        
        print(f"Chatbot: {response}")

# Console demonstration
if __name__ == "__main__":
    # url = input("Enter the website URL: ")
    chatbot()