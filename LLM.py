import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Title for the page
st.title("My First Hugging Face Chatbot")

# Load the model and tokenizer once
@st.cache_resource
def setup_bot_model():
    bot_model_name = "microsoft/DialoGPT-small"  # Small, fast, open-source chat model
    loaded_tokenizer = AutoTokenizer.from_pretrained(bot_model_name)
    loaded_model = AutoModelForCausalLM.from_pretrained(bot_model_name)
    return loaded_tokenizer, loaded_model

chat_tokenizer, chat_model = setup_bot_model()

# To keep chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Show previous messages
for role, message in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Bot:** {message}")

# User input box
user_message = st.text_input("Type your question or message here:")

if user_message:
    # Add user message to chat history
    st.session_state.chat_history.append(("user", user_message))
    
    # Prepare bot input
    all_messages = ""
    for role, text in st.session_state.chat_history:
        if role == "user":
            all_messages += f"{text}\n"
        else:
            all_messages += f"{text}\n"
    
    # Tokenize and generate a response
    input_ids = chat_tokenizer.encode(all_messages + chat_tokenizer.eos_token, return_tensors='pt')
    chat_output = chat_model.generate(input_ids, max_length=1000, pad_token_id=chat_tokenizer.eos_token_id)
    bot_reply = chat_tokenizer.decode(chat_output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    # Add bot reply to chat history
    st.session_state.chat_history.append(("bot", bot_reply))
    st.markdown(f"**Bot:** {bot_reply}")

# Button to clear chat
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.experimental_rerun()
