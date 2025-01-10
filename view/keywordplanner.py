# view/keywordplanner.py

import streamlit as st
from crewai import Agent, Task, LLM
from crewai.tools import BaseTool
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
from textwrap import dedent
import plotly.express as px
import os
import json

# Initialize session state if needed
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.history = []

def initialize_bigquery_client():
    """
    Initialize BigQuery client using credentials.
    """
    try:
        # For local development - load credentials directly
        credentials_info = {
            "type": "service_account",
            "project_id": "is-madt3-6610424015",
            # Add other credential fields here
        }
        
        credentials = service_account.Credentials.from_service_account_info(credentials_info)
        client = bigquery.Client(credentials=credentials, project=credentials.project_id)
        st.success("BigQuery client initialized successfully!")
        return client
    except Exception as e:
        st.error(f"Error initializing BigQuery client: {e}")
        return None

class GoogleBigQueryTool(BaseTool):
    def __init__(self, client):
        super().__init__(
            name="Google BigQuery Keyword Tool",
            description="Fetches keyword data from BigQuery database with monthly searches and competition data",
        )
        self._client = client

    def _run(self, query: str) -> str:
        """Execute the tool's main functionality."""
        if not self._client:
            return "BigQuery client not initialized."
        try:
            df = self._execute_query(query)
            if df.empty:
                return "No results found for the given keyword."
            return df.to_json(orient="records")
        except Exception as e:
            return f"Error executing query: {str(e)}"

    def _execute_query(self, keyword: str) -> pd.DataFrame:
        """Execute BigQuery query with proper parameter handling."""
        if not self._client:
            raise Exception("BigQuery client not initialized.")
        try:
            query = """
            SELECT
                keyword,
                avg_monthly_searches,
                competition
            FROM `is-madt3-6610424015.is_kw_dataset.cus_kw`
            WHERE 
                LOWER(keyword) LIKE LOWER(@keyword_pattern)
                AND avg_monthly_searches IS NOT NULL
            ORDER BY avg_monthly_searches DESC
            LIMIT 100
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("keyword_pattern", "STRING", f"%{keyword}%"),
                ]
            )
            df = self._client.query(query, job_config=job_config).result().to_dataframe()
            return df
        except Exception as e:
            st.error(f"Query execution error: {e}")
            return pd.DataFrame()

    def execute_query(self, keyword: str) -> pd.DataFrame:
        """Public method for executing queries."""
        return self._execute_query(keyword)

class KeywordPlannerAgent:
    def __init__(self, gemini_api_key, bigquery_client):
        """Initialize Keyword Planner agents."""
        self.llm = LLM(model="gemini/gemini-1.5-pro-latest", 
                       api_key=gemini_api_key)
        self.bigquery_tool = GoogleBigQueryTool(bigquery_client)

        self.keyword_researcher = Agent(
            role="Keyword Research Specialist",
            goal="Research and analyze keywords to discover high-potential opportunities",
            backstory=dedent("""
                You're a seasoned keyword research specialist with expertise in:
                - Analyzing search volumes and competition metrics
                - Identifying valuable keyword opportunities
                - Understanding search intent and user behavior
                """),
            tools=[self.bigquery_tool],
            llm=self.llm,
            verbose=True
        )

    def create_research_task(self, query_input: str) -> Task:
        return Task(
            description=dedent(f"""
                Perform keyword discovery and analysis for: {query_input}
                
                Steps:
                1. Query BigQuery for comprehensive keyword data
                2. Analyze search volumes and competition metrics
                3. Identify high-potential keywords
                4. Filter and rank based on performance indicators
                """),
            expected_output="Detailed keyword analysis with metrics and insights",
            agent=self.keyword_researcher
        )

def run_keywordplanner_agent():
    """Main function to run the Keyword Planner interface."""
    
    st.title("🔍 Keyword Planner")
    st.write("Analyze keywords and develop content strategies using AI-powered analysis.")

    # Initialize session state for chat messages
    if "kw_messages" not in st.session_state:
        st.session_state.kw_messages = []

    # Clear chat button
    if st.button('Clear Analysis History', type="primary"):
        st.session_state.kw_messages.clear()

    # Create input form
    with st.form("keyword_analysis_form"):
        keyword_input = st.text_input(
            "Enter Keyword or Topic",
            help="Enter your main keyword or topic to analyze"
        )
        analyze_button = st.form_submit_button("Analyze Keywords")

    if analyze_button and keyword_input:
        # Initialize BigQuery client
        client = initialize_bigquery_client()
        if not client:
            return

        try:
            with st.spinner("Analyzing keywords... Please wait."):
                planner = KeywordPlannerAgent(st.secrets["GEMINI_API_KEY"], client)
                df = planner.bigquery_tool.execute_query(keyword_input)
                
                if df.empty:
                    st.warning("No results found for your query.")
                    return

                # Display results
                st.subheader("Keyword Analysis Results")
                st.dataframe(df, use_container_width=True)

                # Create visualization for top keywords
                st.subheader("Top Keywords by Search Volume")
                fig = px.bar(
                    df.head(10),
                    x="keyword",
                    y="avg_monthly_searches",
                    title="Top 10 Keywords by Monthly Searches",
                    labels={"keyword": "Keyword", "avg_monthly_searches": "Monthly Searches"}
                )
                fig.update_layout(xaxis_tickangle=-45, height=500)
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")

if __name__ == "__main__":
    run_keywordplanner_agent()
