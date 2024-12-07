import requests
from bs4 import BeautifulSoup
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer
import warnings
warnings.filterwarnings("ignore")

# Hugging Face API URL and model
HF_API_URL = "https://api-inference.huggingface.co/models/gpt2"
HF_HEADERS = {"Authorization": "Bearer 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"}  # Your API key

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load a SentenceTransformer model for embedding the website content
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Function to send a request to Hugging Face for text generation
def chat_with_hf_model(prompt, max_tokens=100):
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_tokens}  # Limit the number of tokens the model can generate
    }
    response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload)
    
    if response.status_code == 200:
        response_data = json.loads(response.content)
        return response_data[0]['generated_text']
    else:
        print(f"Error: {response.status_code} - {response.content.decode()}")
        return "Sorry, I couldn't generate a response."

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

# Function to dynamically truncate content based on token limits
def truncate_content(content, max_tokens=100):
    # Tokenize the prompt
    tokens = tokenizer.encode(content)
    
    # If the tokenized prompt is larger than the allowed max tokens, truncate it
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        # Decode the truncated tokens back to text
        truncated_prompt = tokenizer.decode(truncated_tokens)
        return truncated_prompt
    return content

# Function to find relevant sections of the content based on user input
# def find_relevant_content(content, user_input):
#     # Simple approach: split content into paragraphs and find ones containing keywords from the user input
#     paragraphs = content.split('\n')
#     relevant_paragraphs = [para for para in paragraphs if any(word in para for word in user_input.split())]
    
#     # Join the relevant paragraphs, or if none are found, fallback to a short summary
#     if relevant_paragraphs:
#         return " ".join(relevant_paragraphs)
#     else:
#         # Fallback: summarize the entire content (or provide the first 500 characters as a simple summary)
#         return content[:500]
    
# Main chatbot function
def chatbot(url= 'https://botpenguin.com/'):
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
        
        # Find relevant sections of the website content based on the user input
        relevant_chunks = retrieve_relevant_chunks(user_input, content_chunks, index, embeddings, top_k=2)
        
        # Join the relevant chunks
        relevant_content = " ".join(relevant_chunks)
        # relevant_content = find_relevant_content(website_content, user_input)
        
        # Truncate the relevant content to fit within the token limit
        relevant_content = truncate_content(relevant_content, max_tokens=700)  # Reserve space for user input
        # Prepare the prompt for the text generation model
        prompt = f"Using the following information from the website:\n{relevant_content}\nUser: {user_input}"
        
        # Truncate the prompt dynamically so it fits within the token limit
        prompt = truncate_content(prompt, max_tokens=700)  # Limit input tokens to 900, leaving 100 for generation
        
        # Generate the response with a max of 100 tokens
        response = chat_with_hf_model(prompt, max_tokens=100)
        
        print(f"Chatbot: {response}")
        
# Console demonstration
if __name__ == "__main__":
    # url = input("Enter the website URL: ")
    chatbot()