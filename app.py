import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType, Tool

from langchain_community.utilities import (
    WikipediaAPIWrapper,
    ArxivAPIWrapper,
)

load_dotenv()

# ---------------- Page ----------------

st.set_page_config(
    page_title="LangSearch Chatbot",
    page_icon="🔍"
)

st.title("🔍 LangSearch Chatbot")

# ---------------- Sidebar ----------------

st.sidebar.title("Settings")
with st.sidebar.form("api_key_form"):
    api_key_input = st.text_input(
        "Enter your Groq API Key:",
        type="password",
        value=st.session_state.get("api_key", "")
    )
    submitted = st.form_submit_button("Save Key")
    if submitted:
        st.session_state.api_key = api_key_input

api_key = st.session_state.get("api_key", "")

# ---------------- Tools ----------------

wiki_wrapper = WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=300
)


def safe_wikipedia_search(query: str) -> str:
    try:
        return wiki_wrapper.run(query)
    except Exception as e:
        return f"Wikipedia search failed: {e}. Try rephrasing the query."


wikipedia = Tool(
    name="wikipedia",
    func=safe_wikipedia_search,
    description=(
        "Search Wikipedia for general knowledge and definitions. "
        "Input should be a search query string."
    )
)

arxiv_wrapper = ArxivAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=300,
    load_max_docs=1,
)


def safe_arxiv_search(query: str) -> str:
    
    try:
        return arxiv_wrapper.run(query)
    except Exception as e:
        return (
            "Arxiv search is currently unavailable (likely rate-limited, "
            f"HTTP 429): {e}. Please answer using Wikipedia results instead, "
            "or mention that Arxiv could not be reached right now."
        )


arxiv = Tool(
    name="arxiv",
    func=safe_arxiv_search,
    description=(
        "Search Arxiv.org for scientific papers. Input should be a "
        "search query string."
    )
)

tools = [wikipedia, arxiv]

# ---------------- Chat History ----------------

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi, I'm a chatbot who can search Wikipedia and Arxiv. How can I help you?"
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ---------------- User Input ----------------

prompt = st.chat_input("Ask anything...")

if prompt:

    if not api_key:
        st.warning("Please enter your Groq API Key.")
        st.stop()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.write(prompt)
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        streaming=False
    )

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=4,
        max_execution_time=30,
        early_stopping_method="generate",
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = agent.run(prompt)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response
                    }
                )

                st.write(response)

            except Exception as e:
                st.error(
                    "Something went wrong. If this is due to an Arxiv/"
                    "Wikipedia rate limit (HTTP 429), please try again "
                    "in a little while.\n\nDetails: " + str(e)
                )