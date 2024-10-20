import os
import json
from dotenv import load_dotenv
import streamlit as st
import requests

# Step 1: Load environment variables from .env
load_dotenv()

# Step 2: Retrieve the NVIDIA API key from the environment variables
def get_nvidia_api_key():
    """Retrieve NVIDIA API key from environment variable."""
    nvidia_api_key = os.getenv("NVIDIA_API_KEY")
    if not nvidia_api_key:
        raise ValueError("NVIDIA_API_KEY environment variable is not set")
    return nvidia_api_key

# Configuration
MODEL = "nvidia/llama-3.1-nemotron-70b-instruct"
TEMPERATURE = 0.5
TOP_P = 1
MAX_TOKENS = 1024
API_BASE = "https://integrate.api.nvidia.com/v1"

def generate_response(user_input, conversation_history, api_key):
    """Generate response from NVIDIA LLaMA API."""
    try:
        url = f"{API_BASE}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": MODEL,
            "messages": [{"role": "user", "content": user_input}],
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": MAX_TOKENS
        }

        response = requests.post(url, headers=headers, json=data)

        # Debugging: Log request and response details (Remove in production)
        # print(f"Request URL: {url}")
        # print(f"Request Headers: {headers}")
        # print(f"Request Body: {data}")
        # print(f"Response Status Code: {response.status_code}")
        # print(f"Response Body: {response.text}")

        response.raise_for_status()  # Raise an exception for HTTP errors

        response_json = response.json()
        response_text = response_json['choices'][0]['message']['content']
        return response_text

    except requests.exceptions.HTTPError as http_err:
        return f"HTTP Error: {http_err}"
    except Exception as err:
        return f"Error: {err}"

def validate_input(user_input):
    """Implement your validation logic here (e.g., check for specific keywords, length, etc.)."""
    # Example: Check for empty input
    if not user_input.strip():
        return False
    return True

def send_message(api_key):
    user_input = st.session_state.user_input
    if user_input.lower() == 'quit':
        st.session_state.conversation_history = "Goodbye!"
        st.session_state.user_input_history = []
    elif validate_input(user_input):
        st.session_state.user_input_history.append("User: " + user_input)
        response = generate_response(user_input, "\n".join(st.session_state.user_input_history), api_key)
        st.session_state.user_input_history.append("LLaMA: " + response)
        st.session_state.conversation_history = "\n".join(st.session_state.user_input_history)

        if response.startswith("Error") or response.startswith("HTTP Error"):
            st.error(response)
        else:
            st.write("LLaMA:", response)

    # Clear the input field
    st.session_state.user_input = ""

def main():
    nvidia_api_key = get_nvidia_api_key()

    # Initialize Session State for Conversation History
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = ""
    if 'user_input_history' not in st.session_state:
        st.session_state.user_input_history = []
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""

    # Streamlit Interface
    st.title("NVIDIA LLaMA Chatbot")
    st.write("Welcome to the NVIDIA LLaMA Chatbot! Type 'quit' to exit the conversation.")

    # Input Text Box
    user_input = st.text_input("You:", value="", key="user_input")

    # Send Button with Callback
    st.button("Send", on_click=send_message, args=(nvidia_api_key,))

    # Display Conversation History
    st.write("**Conversation History:**")
    st.write(st.session_state.conversation_history)

    # Security Improvement: Limit Conversation History Size
    if len(st.session_state.user_input_history) > 20:
        st.session_state.user_input_history = st.session_state.user_input_history[-20:]
        st.session_state.conversation_history = "\n".join(st.session_state.user_input_history)

if __name__ == "__main__":
    main()