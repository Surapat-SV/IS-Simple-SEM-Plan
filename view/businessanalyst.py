__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
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

class BusinessAnalystChatbot:
    QUESTIONS = [
        "What is the name of your business?",
        "What products or services do you offer?",
        "Who is your target audience?",
        "What makes your product/service unique?",
        "What are your key marketing goals?"
    ]

    def __init__(self):
        self.current_question_index = 0

    def get_next_question(self):
        if self.current_question_index < len(self.QUESTIONS):
            question = self.QUESTIONS[self.current_question_index]
            self.current_question_index += 1
            return question
        return None

    @staticmethod
    def create_agent():
        return Agent(
            role="Business Analyst",
            goal="Assist in collecting detailed business information through structured conversation.",
            backstory="An experienced business analyst skilled in gathering and structuring business data for strategic purposes.",
            verbose=True,
            allow_delegation=False
        )

    @staticmethod
    def create_task(agent, context):
        return Task(
            description="Engage in a structured conversation to collect business details.",
            agent=agent,
            expected_output="Detailed business information in JSON format.",
            context=context
        )

def run_business_analyst_chatbot():
    st.title("Business Analyst Chatbot")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "context" not in st.session_state:
        st.session_state["context"] = {}
    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = str(uuid.uuid4())
    if "current_question_index" not in st.session_state:
        st.session_state["current_question_index"] = 0

    chatbot = BusinessAnalystChatbot()

    # Greeting
    if not st.session_state["messages"]:
        greeting = "Hello! I'm here to assist you with your business. Let's get started!"
        st.session_state["messages"].append({"role": "assistant", "content": greeting})
        st.chat_message("assistant").write(greeting)

    # Display chat history
    for msg in st.session_state["messages"]:
        avatar = "👤" if msg["role"] == "user" else "🤖"
        st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

    # User input
    user_input = st.chat_input(placeholder="Your answer here...")
    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # Save user response to context
        current_question_index = st.session_state["current_question_index"]
        question_key = f"question_{current_question_index}"
        st.session_state["context"][question_key] = user_input

        # Get next question
        next_question = chatbot.get_next_question()
        if next_question:
            st.session_state["messages"].append({"role": "assistant", "content": next_question})
            st.chat_message("assistant").write(next_question)
        else:
            # Create agent and task
            agent = chatbot.create_agent()
            task = chatbot.create_task(agent, st.session_state["context"])

            # Create crew and execute task
            try:
                handler = StreamlitCallbackHandler("Business Analyst")
                crew = Crew(agents=[agent], tasks=[task], verbose=True, callbacks=[handler], process=Process.sequential)
                result = crew.kickoff()

                # Append result to messages
                final_output = result["output"] if isinstance(result, dict) and "output" in result else "Thank you for your responses!"
                st.session_state["messages"].append({"role": "assistant", "content": final_output})
                st.chat_message("assistant").write(final_output)
            except Exception as e:
                st.session_state["messages"].append({"role": "assistant", "content": f"An error occurred: {str(e)}"})
                st.chat_message("assistant").write(f"An error occurred: {str(e)}")

    # Clear chat history button
    if st.button("Clear Chat History"):
        st.session_state.clear()
        st.rerun()

if __name__ == "__main__":
    run_business_analyst_chatbot()
