import requests
from bs4 import BeautifulSoup
import json

# Hugging Face API URL and model
HF_API_URL = "https://api-inference.huggingface.co/models/gpt2"
HF_HEADERS = {"Authorization": "Bearer #####Your API_key from Hugging_Face####"}  # You can get an API key from Hugging Face, some models are free to use

# Function to send a request to Hugging Face for text generation
def chat_with_hf_model(prompt):
    payload = {"inputs": prompt}
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
        # print(content)
    # exit()
    return content

# Main chatbot function
def chatbot(url):
    # Fetch the website data
    soup = fetch_website_content(url)
    if not soup:
        return
    
    # Extract and process key data
    website_content = extract_key_information(soup)
    print(f"Website content fetched and processed.\n")

    # User interaction loop
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Exiting the chatbot.")
            break
        
        # Prepare the prompt for the text generation model
        prompt = f"Using the following information from the website:\n{website_content}\nUser: {user_input}"
        response = chat_with_hf_model(prompt)
        
        print(f"Chatbot: {response}")

# Console demonstration
if __name__ == "__main__":
    url = input("Enter the website URL: ")
    chatbot(url)
