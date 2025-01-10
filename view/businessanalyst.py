__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from dotenv import load_dotenv
import json
import pandas as pd
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
    """Class to handle business research and data collection"""
    def __init__(self):
        self.serper_api_key = st.secrets['SERPER_API_KEY']
        self.gemini_api_key = st.secrets['GEMINI_API_KEY']
        self.llm = LLM(
            model="gemini/gemini-1.5-pro-latest",
            api_key=self.gemini_api_key,
            temperature=0
        )
        self.search_tool = SerperDevTool(api_key=self.serper_api_key)
        self.scrape_tool = ScrapeWebsiteTool()

    def create_researcher_agent(self):
        return Agent(
            role="Business Researcher",
            goal="Engage users in structured conversation to gather detailed business information.",
            backstory="Expert in business research and data collection",
            tools=[self.search_tool, self.scrape_tool],
            allow_delegation=False,
            llm=self.llm
        )

    def create_research_task(self, agent, context):
        return Task(
            description=f"""
            Engage in a structured conversation to collect business data:
            - Context: {context}
            """,
            expected_output="Structured JSON containing business details such as name, products/services, target audience, and goals.",
            agent=agent
        )

def run_businessanalyst():
    """Run the Business Analyst chatbot"""
    st.title("Business Analyst Chatbot")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "context" not in st.session_state:
        st.session_state["context"] = {}
    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = str(uuid.uuid4())

    # Display chat history
    for msg in st.session_state["messages"]:
        avatar = "👤" if msg["role"] == "user" else "🤖"
        st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

    # User input
    user_input = st.chat_input(placeholder="Tell me about your business...")
    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        try:
            # Initialize researcher
            researcher = BusinessResearcher()
            agent = researcher.create_researcher_agent()
            task = researcher.create_research_task(agent, st.session_state["context"])

            # Create Crew and execute task
            handler = StreamlitCallbackHandler("Business Analyst")
            crew = Crew(
                agents=[agent],
                tasks=[task],
                verbose=True,
                process=Process.sequential,
                callbacks=[handler]
            )
            results = crew.kickoff()

            # Process results
            if isinstance(results, list) and results:
                final_result = results[-1]
            else:
                final_result = str(results)

            # Update session state with results
            st.session_state["messages"].append({"role": "assistant", "content": final_result})
            st.chat_message("assistant").write(final_result)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # Clear chat history button
    if st.button("Clear Chat History"):
        st.session_state.clear()
        st.rerun()

if __name__ == "__main__":
    run_businessanalyst()
