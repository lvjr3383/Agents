import streamlit as st
from agents import SwooshAgent
from openai import OpenAI
import json
from dotenv import load_dotenv
load_dotenv()

# Load shoes data
def load_shoes():
    with open("shoes.json") as f:
        return json.load(f)

shoes_data = load_shoes()
client = OpenAI()

# Initialize agent
swoosh = SwooshAgent(client, shoes_data)

# Streamlit App
st.set_page_config(page_title="Nike AI", layout="centered")
st.markdown("""
<style>
    .main {background-color: #f5_Black; color: white;}
    .stButton>button {background-color: #FF6900; color: white; border-radius: 50px;}
    .card {background: white; color: black; padding: 20px; border-radius: 15px; margin: 10px;}
</style>
""", unsafe_allow_html=True)

st.image("https://upload.wikimedia.org/wikipedia/commons/a/a6/Logo_NIKE.svg", width=80)
st.title("Nike AI — Just Do It")

if 'messages' not in st.session_state:
    st.session_state.messages = []
    intro = swoosh.handle_input({"name": "Jack", "member_since": 2011})
    st.session_state.messages.append({"role": "assistant", "content": intro["output"]})

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask Swoosh..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call agent
    response = swoosh.handle_input({"name": "Jack"}, prompt)
    st.session_state.messages.append({"role": "assistant", "content": response["output"]})
    with st.chat_message("assistant"):
        st.markdown(response["output"])