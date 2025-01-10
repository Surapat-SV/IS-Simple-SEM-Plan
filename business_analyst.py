__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import json
from crewai import Crew, Agent, Task
from langchain_core.callbacks import BaseCallbackHandler
import streamlit as st
import uuid

class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, agent_name: str):
        self.agent_name = agent_name

    def on_chain_start(self, serialized, inputs, **kwargs):
        # Show the input of the current chain in the chat
        message = inputs.get("input", "Processing...")
        if "messages" in st.session_state:
            st.session_state["messages"].append({"role": self.agent_name, "content": message})
        st.chat_message("assistant").write(message)

    def on_chain_end(self, outputs, **kwargs):
        # Extract the natural language response from the output
        if "tasks_output" in outputs and outputs["tasks_output"]:
            task_output = outputs["tasks_output"][0]  # Assuming the first task is relevant
            message = task_output.get("raw", "Completed")  # Extract the natural response
        else:
            message = outputs.get("output", "Completed")

        if "messages" in st.session_state:
            st.session_state["messages"].append({"role": self.agent_name, "content": message})
        st.chat_message("assistant").write(message)


class BusinessAnalystAgents:
    @staticmethod
    def conversational_business_analyst_agent():
        return Agent(
            role="Conversational Business Analyst",
            goal="Gather business requirements through structured conversation",
            backstory=(
                "Expert Business Analyst skilled in requirement gathering. "
                "Asks questions one at a time and builds upon previous answers."
            ),
            verbose=True,
            memory=True,
            allow_delegation=False
        )

class BusinessAnalystTasks:
    def __init__(self):
        self.questions = [
            "What is the name of your business?",
            "What products or services do you offer?",
            "Who is your target audience?",
            "What makes your product/service unique?",
            "What are your key marketing goals?"
        ]
        
    def get_current_question_index(self, context):
        for i in range(len(self.questions)):
            if f"answer_{i}" not in context:
                return i
        return len(self.questions)

    def format_context_summary(self, context):
        summary = {}
        for i in range(len(self.questions)):
            key = f"answer_{i}"
            if key in context:
                summary[self.questions[i]] = context[key]
        return summary

    def conversational_gathering_task(self, agent, context, user_input):
        # Initialize or update context
        if user_input and not context:
            # First message - just store it and start questions
            context["initial_message"] = user_input
            current_index = 0
        else:
            # Get current question index
            current_index = self.get_current_question_index(context)
            # Store the answer for the previous question if applicable
            if current_index > 0 and user_input:
                prev_index = current_index - 1
                context[f"answer_{prev_index}"] = user_input

        # Check if we've completed all questions
        if current_index >= len(self.questions):
            summary = self.format_context_summary(context)
            return Task(
                description=f"""
                All questions have been answered. Here's the summary:
                {json.dumps(summary, indent=2)}
                
                Please provide a comprehensive summary and next steps.
                """,
                expected_output="Final summary and recommendations",
                agent=agent
            )

        # Format context for current question
        context_summary = self.format_context_summary(context)
        context_str = json.dumps(context_summary, indent=2) if context_summary else "No previous answers yet"

        return Task(
            description=f"""
            Current question to ask: {self.questions[current_index]}
            
            Previous context:
            {context_str}
            
            Latest user input: {user_input if user_input else 'None'}
            
            Instructions:
            1. If there was a previous answer, acknowledge it briefly
            2. Ask the current question clearly and politely
            3. Maintain a natural conversational flow
            """,
            expected_output="Natural conversation gathering business requirements",
            agent=agent
        )

def run_business_analyst():
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
        # Append user input
        st.session_state["messages"].append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        try:
            # Initialize agent and task
            agent = BusinessAnalystAgents.conversational_business_analyst_agent()
            task = BusinessAnalystTasks().conversational_gathering_task(
                agent,
                st.session_state["context"],
                user_input
            )

            # Create and execute crew
            handler = StreamlitCallbackHandler("Business Analyst")
            crew = Crew(
                agents=[agent],
                tasks=[task],
                verbose=True,
                callbacks=[handler]
            )

            result = crew.kickoff()

            # Update session state
            st.chat_message("assistant").write(result)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # Clear chat history button
    if st.button("Clear Chat History"):
        st.session_state.clear()
        st.rerun()
