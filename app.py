import os
from dotenv import load_dotenv
import streamlit as st
import requests

# Step 1: Load environment variables from .env (for local development)
load_dotenv()

# Step 2: Retrieve the NVIDIA API key
# For Streamlit Cloud, use st.secrets
if "NVIDIA_API_KEY" in st.secrets:
    nvidia_api_key = st.secrets["NVIDIA_API_KEY"]
elif os.getenv("NVIDIA_API_KEY"):
    nvidia_api_key = os.getenv("NVIDIA_API_KEY")  # Fallback for local development
else:
    raise ValueError("NVIDIA_API_KEY environment variable is not set")

# Configuration
model = "nvidia/llama-3.1-nemotron-70b-instruct"
temperature = 0.5
top_p = 1
max_tokens = 1024
api_base = "https://integrate.api.nvidia.com/v1"

def is_response_complete(text):
    """
    Determines if the response text is complete based on its ending punctuation.
    """
    if not text:
        return True  # Empty response can be considered complete
    return text.strip().endswith(('.', '!', '?'))

def generate_response(user_input, conversation_history):
    try:
        url = f"{api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {nvidia_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": [{"role": "user", "content": user_input}],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens
        }

        response = requests.post(url, headers=headers, json=data)

        # Debugging: Print request and response details (Remove in production)
        print(f"Request URL: {url}")
        print(f"Request Headers: {headers}")
        print(f"Request Body: {data}")
        print(f"Response Status Code: {response.status_code}")
        print(f"Response Body: {response.text}")

        if response.status_code == 200:
            response_json = response.json()
            # Adjust based on NVIDIA's response structure
            response_text = response_json['choices'][0]['message']['content']
            complete = is_response_complete(response_text)
            return response_text, complete
        elif response.status_code == 401:
            return "Authentication Error: Invalid API key or insufficient permissions.", True
        else:
            return f"Error occurred while processing your request: HTTP {response.status_code} - {response.text}", True
    except Exception as e:
        return f"Error occurred while processing your request: {e}", True

def send_continue():
    # Retrieve the last incomplete response
    if not st.session_state.user_input_history:
        st.warning("No previous response to continue.")
        return

    last_entry = st.session_state.user_input_history[-1]
    if not last_entry.startswith("LLaMA: "):
        st.warning("Last entry is not a model response.")
        return

    last_llama_response = last_entry.replace("LLaMA: ", "").strip()

    # Generate a prompt to continue from the last incomplete response
    continuation_prompt = f"Continue the following response:\n\n{last_llama_response}"

    # Append the continuation prompt to the conversation history
    st.session_state.user_input_history.append("User: " + continuation_prompt)

    # Generate the continuation response
    response, complete = generate_response(continuation_prompt, "\n".join(st.session_state.user_input_history))

    # Append the continuation to the conversation history
    st.session_state.user_input_history.append("LLaMA: " + response)
    st.session_state.conversation_history = "\n".join(st.session_state.user_input_history)
    st.session_state.last_response_complete = complete

    # Error Handling: Check if response is an error message
    if response.startswith("Error occurred while processing your request:") or response.startswith("Authentication Error:"):
        st.error(response)
    else:
        st.write("LLaMA:", response)

def validate_input(user_input):
    # Implement your validation logic here (e.g., check for specific keywords, length, etc.)
    return True

def main():
    # Initialize Session State for Conversation History and Completion Flag
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = ""
    if 'user_input_history' not in st.session_state:
        st.session_state.user_input_history = []
    if 'last_response_complete' not in st.session_state:
        st.session_state.last_response_complete = True  # Assume initial state is complete

    # Streamlit Interface
    st.title("NVIDIA LLaMA Chatbot")
    st.write("Welcome to the NVIDIA LLaMA Chatbot! Type 'quit' to exit the conversation.")

    # Using a form to encapsulate input and submission
    with st.form(key='chat_form'):
        user_input = st.text_input("You:", value="", key="user_input")
        submit_button = st.form_submit_button(label='Send')

    # Display Conversation History
    st.write("**Conversation History:**")
    st.write(st.session_state.conversation_history)

    if submit_button and user_input:
        if user_input.lower() == 'quit':
            st.write("Goodbye!")
            st.session_state["conversation_history"] = ""  # Reset conversation history
            st.session_state["user_input_history"] = []  # Reset user input history
            st.session_state["last_response_complete"] = True  # Reset flag
            st.experimental_rerun()
        elif validate_input(user_input):
            # Update Conversation History
            st.session_state.user_input_history.append("User: " + user_input)
            response, complete = generate_response(user_input, "\n".join(st.session_state.user_input_history))
            st.session_state.user_input_history.append("LLaMA: " + response)
            st.session_state.conversation_history = "\n".join(st.session_state.user_input_history)
            st.session_state.last_response_complete = complete

            # Error Handling: Check if response is an error message
            if response.startswith("Error occurred while processing your request:") or response.startswith("Authentication Error:"):
                st.error(response)
            else:
                st.write("LLaMA:", response)

        # Clear the input field by resetting the form
        

    # Display "Continue" Button if Last Response Was Incomplete
    if not st.session_state.last_response_complete:
        if st.button("Continue"):
            send_continue()
            st.experimental_rerun()

    # Security Improvement: Limit Conversation History Size
    if len(st.session_state.user_input_history) > 20:
        st.session_state.user_input_history = st.session_state.user_input_history[-20:]
        st.session_state.conversation_history = "\n".join(st.session_state.user_input_history)

if __name__ == "__main__":
    main()