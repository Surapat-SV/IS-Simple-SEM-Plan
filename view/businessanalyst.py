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
        # Use the current_question_index from session_state instead
        if st.session_state["current_question_index"] < len(self.QUESTIONS):
            question = self.QUESTIONS[st.session_state["current_question_index"]]
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
    # Convert dictionary to a list of key-value pairs
    context_as_list = [{"question": key, "answer": value} for key, value in context.items()]
        return Task(
            description="Engage in a structured conversation to collect business details.",
            agent=agent,
            expected_output="Detailed business information in JSON format.",
            context=context_as_list  # Pass the list to the context
        )
def run_business_analyst_chatbot():
    st.title("Business Analyst Chatbot")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        # Only add the greeting when initializing messages for the first time
        greeting = "Hello! My name is K'Bus. Can you tell me your Business Name?"
        st.session_state["messages"].append({"role": "assistant", "content": greeting})
    if "context" not in st.session_state:
        st.session_state["context"] = {}
    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = str(uuid.uuid4())
    if "current_question_index" not in st.session_state:
        st.session_state["current_question_index"] = 0

    chatbot = BusinessAnalystChatbot()

    # Display chat history
    for msg in st.session_state["messages"]:
        avatar = "👤" if msg["role"] == "user" else "🤖"
        st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

    # User input
    user_input = st.chat_input(placeholder="Your answer here...")
    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)
    
        # Save user response and increment question index
        current_question_index = st.session_state["current_question_index"]
        st.session_state["context"][BusinessAnalystChatbot.QUESTIONS[current_question_index]] = user_input
        st.session_state["current_question_index"] += 1  # Increment index after saving response
    
        # Get next question only if there are more questions
        if st.session_state["current_question_index"] < len(BusinessAnalystChatbot.QUESTIONS):
            next_question = BusinessAnalystChatbot.QUESTIONS[st.session_state["current_question_index"]]
            st.session_state["messages"].append({"role": "assistant", "content": next_question})
            st.chat_message("assistant").write(next_question)
        else:
            # Handle completion of all questions
            summary = "Thank you for all your responses! Here's a summary:\n"
            for q, a in st.session_state["context"].items():
                summary += f"\n{q}\nAnswer: {a}\n"
            st.session_state["messages"].append({"role": "assistant", "content": summary})
            st.chat_message("assistant").write(summary)
            
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
