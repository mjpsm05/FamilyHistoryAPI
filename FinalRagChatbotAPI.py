from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import requests
import re
import os
from dotenv import load_dotenv

load_dotenv("/Users/mazamessomeba/Desktop/Projects/Streamlit/tokenAndAPI.env")  # Load environment variables from .env file


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Hugging Face API Configuration
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HF_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

# Pinecone API Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = "test"

# Add this after load_dotenv()
print(os.getcwd())
print("Hugging Face API Key:", HUGGINGFACE_API_KEY)
print("Pinecone API Key:", PINECONE_API_KEY)

# Initialize Pinecone (with error checking)
if not PINECONE_API_KEY:
    raise ValueError("Pinecone API key is missing. Check your .env file.")


# Initialize Pinecone
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

# Define index name and specification
if index_name not in pinecone_client.list_indexes().names():
    pinecone_client.create_index(
        name=index_name,
        dimension=384,  # Dimension matches the embedding model output
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Access the index
index = pinecone_client.Index(index_name)

# Initialize the embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Define credentials (Placeholder - In production, use a proper user database)
credentials = {
    "nathaniel": "password1",
    "kwame": "password2",
    "mazamesso": "@Lotovisa05"
}

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username', '')
    password = data.get('password', '')
    
    if username in credentials and credentials[username] == password:
        return jsonify({
            'success': True,
            'username': username,
            'message': f'Welcome {username}!'
        })
    else:
        return jsonify({
            'success': False,
            'message': 'Invalid credentials, please try again.'
        }), 401

@app.route('/query', methods=['POST'])
def process_query():
    data = request.json
    user_query = data.get('query', '')
    username = data.get('username', '')
    
    if not user_query or not username:
        return jsonify({
            'success': False,
            'message': 'Missing query or username'
        }), 400
    
    try:
        # Generate embedding for the query
        query_embedding = embedder.encode(user_query).tolist()
        
        # Query Pinecone for user-specific data
        filter_criteria = {"user": username} if username in credentials else {}
        results = index.query(vector=query_embedding, top_k=3, include_metadata=True, filter=filter_criteria)
        
        # Extract relevant context from documents
        context_list = [match["metadata"].get("text", "") for match in results["matches"] if "text" in match["metadata"]]
        
        if not context_list:
            return jsonify({
                'success': False,
                'message': 'No relevant information found in the document.'
            })
        
        refined_context = " ".join(context_list[:2])  # Use first 2 relevant chunks
        
        # Construct prompt for API
        full_prompt = f"""
        You are a helpful assistant. Based on the context, answer the user's question.

        Context:
        {refined_context}

        Question: {user_query}

        Answer:
        """
        
        # Define headers
        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Send request to Hugging Face API
        response = requests.post(
            HF_API_URL, 
            headers=headers, 
            json={
                "inputs": full_prompt, 
                "parameters": {
                    "max_new_tokens": 1200, 
                    "temperature": 0.5  # Lower temperature for more controlled responses
                }
            }
        )
        
        if response.status_code == 200:
            try:
                ai_response = response.json()[0]["generated_text"]
                
                # Extract the answer using regex
                answer_match = re.search(r"Answer:\s*(.*)", ai_response, re.DOTALL)
                if answer_match:
                    answer = answer_match.group(1).strip()
                else:
                    # Fallback method if regex fails
                    answer_start = ai_response.find("Answer:")
                    answer = ai_response[answer_start + len("Answer:"):].strip() if answer_start != -1 else "Error: Could not extract answer."
                
                return jsonify({
                    'success': True,
                    'response': answer
                })
                
            except (KeyError, IndexError) as e:
                return jsonify({
                    'success': False,
                    'message': f'Unexpected response format: {str(e)}'
                }), 500
        else:
            return jsonify({
                'success': False,
                'message': f'Error generating response: {response.text}'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500

if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=5001)