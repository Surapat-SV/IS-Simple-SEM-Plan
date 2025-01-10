__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv
import json
import uuid

# Load environment variables
load_dotenv()

# Callback Handler for Streamlit
class StreamlitCallbackHandler:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name

    def on_chain_start(self, serialized, inputs, **kwargs):
        message = inputs.get("input", "Processing...")
        if "messages" in st.session_state:
            st.session_state["messages"].append({"role": self.agent_name, "content": message})
        st.chat_message(self.agent_name).write(message)

    def on_chain_end(self, outputs, **kwargs):
        message = outputs.get("output", "Completed")
        if "messages" in st.session_state:
            st.session_state["messages"].append({"role": self.agent_name, "content": message})
        st.chat_message(self.agent_name).write(message)

class BusinessResearcher:
    """Class to handle business questions"""
    def __init__(self):
        self.gemini_api_key = st.secrets['GEMINI_API_KEY']
        self.llm = LLM(
            model="gemini/gemini-1.5-pro-latest",
            api_key=self.gemini_api_key,
            temperature=0
        )

    def create_question_agent(self):
        return Agent(
            role="Business Questioner",
            goal="Ask structured questions to gather business information from the user one by one.",
            backstory="An experienced business analyst that excels in gathering detailed business information through questions.",
            allow_delegation=False,
            llm=self.llm
        )

    def create_question_task(self, agent, question):
        return Task(
            description=f"Ask the user: {question}",
            expected_output=f"The user's response to the question: {question}",
            agent=agent
        )

def run_businessanalyst():
    """Run the Business Analyst chatbot to ask 5 structured questions one by one"""
    st.title("Business Analyst Chatbot")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "current_question" not in st.session_state:
        st.session_state["current_question"] = 0
    if "responses" not in st.session_state:
        st.session_state["responses"] = {}

    questions = [
        "What is the name of your business?",
        "What products or services do you offer?",
        "Who is your target audience?",
        "What makes your product/service unique?",
        "What are your key marketing goals?"
    ]

    # Display chat history
    for msg in st.session_state["messages"]:
        avatar = "👤" if msg["role"] == "user" else "🤖"
        st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

    # Check if all questions have been answered
    if st.session_state["current_question"] < len(questions):
        current_question = questions[st.session_state["current_question"]]

        # Ask the current question
        st.chat_message("assistant").write(current_question)

        # User input
        user_input = st.chat_input(placeholder="Type your answer here...")
        if user_input:
            # Save the user's response
            st.session_state["responses"][current_question] = user_input
            st.session_state["messages"].append({"role": "user", "content": user_input})
            st.chat_message("user").write(user_input)

            # Move to the next question
            st.session_state["current_question"] += 1

    else:
        # Display all collected responses
        st.subheader("Collected Business Information")
        for question, response in st.session_state["responses"].items():
            st.write(f"**{question}**: {response}")

    # Clear chat history button
    if st.button("Clear Chat History"):
        st.session_state.clear()
        st.rerun()

if __name__ == "__main__":
    run_businessanalyst()
