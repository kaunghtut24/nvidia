import os
from dotenv import load_dotenv
import streamlit as st
import requests
import logging
from PIL import Image
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env (for local development)
load_dotenv()

# Retrieve the NVIDIA API key
nvidia_api_key = st.secrets.get("NVIDIA_API_KEY") or os.getenv("NVIDIA_API_KEY")
if not nvidia_api_key:
    st.error("NVIDIA_API_KEY environment variable is not set")
    st.stop()

# Configuration
model = "nvidia/llama-3.1-nemotron-70b-instruct"
temperature = 0.5
top_p = 1
max_tokens = 1024
api_base = "https://integrate.api.nvidia.com/v1"

# Function to load images from a URL
def load_image_from_url(url, width=50):
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        image = image.resize((width, width))
        return image
    except Exception as e:
        logging.error(f"Error loading image from {url}: {e}")
        return None

# Replace with your actual image URLs
logo_url = "https://www.itworldcanada.com/ai/wp-content/uploads/2018/06/Wcem1g-S_400x400-1.jpg"  # Replace with your logo URL
user_icon_url = "https://cdn-icons-png.flaticon.com/512/3686/3686930.png"  # Example user icon
llm_icon_url = "https://cdn-icons-png.flaticon.com/256/10645/10645125.png"  # Replace with your LLM icon URL

# Load the images
logo_image = load_image_from_url(logo_url, width=150)  # Adjust width as needed
user_icon = load_image_from_url(user_icon_url)
llm_icon = load_image_from_url(llm_icon_url)

# Function to display messages with icons
def display_message(role, content):
    """
    Displays a message with an icon based on the role.
    :param role: 'user' or 'assistant'
    :param content: The message content
    """
    if role == 'user':
        icon = user_icon
        label = "**You**"
    elif role == 'assistant':
        icon = llm_icon
        label = "**LLaMA**"
    else:
        icon = None
        label = f"**{role.capitalize()}**"

    if icon:
        col1, col2 = st.columns([1, 10])
        with col1:
            st.image(icon, use_column_width=True)
        with col2:
            st.markdown(f"{label}: {content}")
    else:
        st.markdown(f"{label}: {content}")

# Function to check if the response is complete
def is_response_complete(text):
    """
    Determines if the response text is complete based on its ending punctuation.
    """
    if not text:
        return True  # Empty response can be considered complete
    return text.strip().endswith(('.', '!', '?'))

# Function to generate a response from the API
def generate_response(conversation_messages):
    try:
        url = f"{api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {nvidia_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": conversation_messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens
        }

        response = requests.post(url, headers=headers, json=data)

        # Log request and response details
        logging.debug(f"Request URL: {url}")
        logging.debug(f"Request Headers: {headers}")
        logging.debug(f"Request Body: {data}")
        logging.debug(f"Response Status Code: {response.status_code}")
        logging.debug(f"Response Body: {response.text}")

        if response.status_code == 200:
            response_json = response.json()
            # Adjust based on NVIDIA's response structure
            response_text = response_json['choices'][0]['message']['content']
            complete = is_response_complete(response_text)
            return response_text, complete
        elif response.status_code == 401:
            logging.error(f"Authentication Error: HTTP {response.status_code} - {response.text}")
            return "Authentication Error: Invalid API key or insufficient permissions.", True
        else:
            logging.error(f"Error occurred: HTTP {response.status_code} - {response.text}")
            return "An error occurred while processing your request.", True
    except Exception as e:
        logging.exception("Exception occurred during generate_response")
        return "An unexpected error occurred while processing your request.", True

# Function to handle continuation of incomplete responses
def send_continue():
    # Retrieve the last incomplete response
    messages = st.session_state.conversation_state['messages']
    if not messages:
        st.warning("No previous response to continue.")
        return

    # Find the last assistant's response
    last_assistant_messages = [msg for msg in reversed(messages) if msg['role'] == 'assistant']
    if not last_assistant_messages:
        st.warning("No assistant response to continue from.")
        return

    last_assistant_message = last_assistant_messages[0]['content']

    # Generate a prompt to continue from the last incomplete response
    continuation_prompt = f"Please continue the following response:\n\n{last_assistant_message}"

    # Append the continuation prompt to the conversation history
    messages.append({'role': 'user', 'content': continuation_prompt})

    # Generate the continuation response
    response, complete = generate_response(messages)

    # Append the continuation to the conversation history
    messages.append({'role': 'assistant', 'content': response})
    st.session_state.conversation_state['last_response_complete'] = complete

    # Error Handling: Check if response is an error message
    if "error occurred" in response.lower() or "authentication error" in response.lower():
        st.error(response)
    else:
        display_message("assistant", response)

# Function to validate user input
def validate_input(user_input):
    # Validation logic for user input
    if len(user_input.strip()) == 0:
        st.warning("Input cannot be empty.")
        return False
    elif len(user_input) > 1000:
        st.warning("Input is too long. Please limit your message to 1000 characters.")
        return False
    return True

# Function to handle quitting the conversation
def _handle_quit_conversation():
    """Reset conversation state and display goodbye message"""
    st.write("Goodbye!")
    st.session_state.conversation_state = {
        'messages': [],
        'last_response_complete': True
    }
    st.experimental_rerun()  # Ensure your Streamlit version supports this

# Function to clear the conversation
def _clear_conversation():
    """Reset conversation state"""
    st.session_state.conversation_state = {
        'messages': [],
        'last_response_complete': True
    }
    st.experimental_rerun()

# Function to handle user input
def _handle_user_input(user_input):
    """Update conversation state, generate response, and display it"""
    # Update Conversation History
    st.session_state.conversation_state['messages'].append({'role': 'user', 'content': user_input})
    # Generate the response
    response, complete = generate_response(st.session_state.conversation_state['messages'])
    # Update conversation history with assistant's reply
    st.session_state.conversation_state['messages'].append({'role': 'assistant', 'content': response})
    st.session_state.conversation_state['last_response_complete'] = complete

    # Error Handling: Check if response is an error message
    if "error occurred" in response.lower() or "authentication error" in response.lower():
        st.error(response)
    else:
        display_message("assistant", response)

    # Security Improvement: Limit Conversation History Size
    if len(st.session_state.conversation_state['messages']) > 20:
        st.session_state.conversation_state['messages'] = st.session_state.conversation_state['messages'][-20:]

# Main function to run the app
def main():
    # Initialize Session State for Conversation State
    if 'conversation_state' not in st.session_state:
        st.session_state.conversation_state = {
            'messages': [],
            'last_response_complete': True
        }

    # Application Header with Logo (Remote Image from URL)
    if logo_image:
        st.sidebar.image(logo_image, use_column_width=True)
    else:
        st.sidebar.write("Logo could not be loaded.")

    # **Adding the Model Description Below the Logo**
    model_description = """
    **NVIDIA LLaMA 3.1 Nemotron-70B Instruct**  
    A state-of-the-art large language model developed by NVIDIA, featuring 70 billion parameters.  
    Designed for advanced natural language understanding and generation, it provides contextually accurate and instruction-following responses.
    """
    st.sidebar.markdown(model_description)

    # Streamlit Interface
    st.title("NVIDIA LLaMA Chatbot")
    st.write("Welcome to the NVIDIA LLaMA Chatbot! Type 'quit' to exit the conversation.")

    # Create a container for the chat box
    chat_container = st.container()

    with chat_container:
        # Display Conversation History
        st.write("**Conversation History:**")
        if not st.session_state.conversation_state['messages']:
            st.write("The conversation is empty. Start by typing your message below.")
        else:
            for msg in st.session_state.conversation_state['messages']:
                display_message(msg['role'], msg['content'])

    # Chat input and submit button within a form
    with st.form(key='chat_form'):
        user_input = st.text_input("You:", value="", key="user_input")
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        if user_input.lower() == 'quit':
            _handle_quit_conversation()
        elif validate_input(user_input):
            _handle_user_input(user_input)

    # Display "Continue" Button if Last Response Was Incomplete
    if not st.session_state.conversation_state['last_response_complete']:
        if st.button("Continue"):
            send_continue()

    # Option to clear the conversation
    if st.button("Clear Conversation"):
        _clear_conversation()

if __name__ == "__main__":
    main()