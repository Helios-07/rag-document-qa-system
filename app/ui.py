import streamlit as st
import requests

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
.chat-container {
    max-width: 800px;
    margin: auto;
}
.user-msg {
    background-color: #2b313e;
    padding: 10px 15px;
    border-radius: 12px;
    margin: 8px 0;
    color: white;
    text-align: right;
}
.bot-msg {
    background-color: #f1f3f6;
    padding: 10px 15px;
    border-radius: 12px;
    margin: 8px 0;
    color: black;
    text-align: left;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 RAG Chatbot")
st.caption("Ask questions from your documents")

with st.sidebar:
    st.header("⚙️ Settings")
    st.write("Model: OpenAI")
    st.write("Retriever: FAISS + reranker")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages=[]

if "messages" not in st.session_state:
    st.session_state.messages=[]

st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg['role']=='user':
        st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">{msg["content"]}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

query=st.chat_input("Ask your question...")

if query:
    st.session_state.messages.append({'role':'user', 'content':query})

    with st.spinner("Thinking..."):
        response=requests.post("http://127.0.0.1:8000/ask-stream", json={"query": query}, stream=True)
    
    full_response=""
    placeholder=st.empty()

    for chunk in response.iter_content(chunk_size=None):
        if chunk:
            text=chunk.decode('utf-8')
            full_response+=text

            placeholder.markdown(f"<div class='bot-msg'>{full_response}</div>", unsafe_allow_html=True)    
    
    st.session_state.messages.append({'role':'assistant', 'content':full_response})

