# view/keywordplanner.py

import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
from textwrap import dedent
import plotly.express as px
import json
from typing import Optional

class GoogleBigQueryTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="Google BigQuery Keyword Tool",
            description="Fetches keyword data from BigQuery database with monthly searches and competition data",
        )
        try:
            # Get credentials from Streamlit secrets
            credentials_info = json.loads(st.secrets["general"]["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
            credentials = service_account.Credentials.from_service_account_info(credentials_info)
            self._client = bigquery.Client(
                credentials=credentials,
                project=credentials_info["project_id"]
            )
        except Exception as e:
            st.error(f"Failed to initialize BigQuery client: {str(e)}")
            self._client = None

    def _run(self, query: str) -> str:
        """Execute the tool's main functionality"""
        try:
            df = self._execute_query(query)
            if df.empty:
                return "No results found for the given keyword."
            
            # Convert DataFrame to list of dictionaries
            results = df.to_dict('records')
            return json.dumps(results, ensure_ascii=False)
        except Exception as e:
            return f"Error executing query: {str(e)}"

    def _execute_query(self, keyword: str) -> pd.DataFrame:
        """Execute BigQuery query with proper parameter handling"""
        if not self._client:
            raise Exception("BigQuery client not initialized")

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
            st.error(f"Query execution error: {str(e)}")
            return pd.DataFrame()

    def execute_query(self, keyword: str) -> pd.DataFrame:
        """Public method for executing queries (used by the Streamlit interface)"""
        return self._execute_query(keyword)

class KeywordPlannerAgent:
    def __init__(self, gemini_api_key):
        """Initialize Keyword Planner agents"""
        self.llm = LLM(model="gemini/gemini-1.5-pro-latest", 
                       api_key=gemini_api_key)
        self.bigquery_tool = GoogleBigQueryTool()

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
        
        self.competition_analyst = Agent(
            role="Competition Analyst",
            goal="Analyze competition levels and difficulty for keywords",
            backstory=dedent("""
                You're an expert in competitive analysis with skills in:
                - Evaluating keyword difficulty and competition
                - Identifying market gaps and opportunities
                - Providing strategic insights for keyword targeting
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

    def create_competition_task(self, query_input: str) -> Task:
        return Task(
            description=dedent(f"""
                Analyze competition levels for keywords related to: {query_input}
                
                Steps:
                1. Evaluate competition metrics from BigQuery
                2. Assess difficulty levels for each keyword
                3. Identify low-competition opportunities
                4. Provide strategic recommendations
                """),
            expected_output="Competition analysis with strategic insights",
            agent=self.competition_analyst
        )

def run_keywordplanner_agent():
    """Main function to run the Keyword Planner interface"""
    
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
        
        analyze_button = st.form_submit_button(
            "Analyze Keywords",
            type="primary",
            use_container_width=True
        )

    if analyze_button and keyword_input:
        try:
            with st.spinner('Analyzing keywords... Please wait.'):
                # Initialize agent with Gemini API key
                planner = KeywordPlannerAgent(st.secrets["GEMINI_API_KEY"])
                
                # Get keyword data
                df = planner.bigquery_tool.execute_query(keyword_input)
                
                if df.empty:
                    st.warning("No results found for your query.")
                    return

                # Display results
                st.subheader("Keyword Analysis Results")
                
                # Format the DataFrame for display
                display_df = df.copy()
                display_df["avg_monthly_searches"] = display_df["avg_monthly_searches"].apply(
                    lambda x: "{:,}".format(int(x)) if pd.notnull(x) else x
                )
                
                # Create competition mapping
                competition_map = {
                    'HIGH': 'สูง',
                    'MEDIUM': 'ปานกลาง',
                    'LOW': 'ต่ำ'
                }
                display_df['competition'] = display_df['competition'].map(competition_map).fillna('ไม่ระบุ')
                
                # Display data table
                st.dataframe(
                    display_df,
                    column_config={
                        "keyword": st.column_config.TextColumn("Keyword", width="medium"),
                        "avg_monthly_searches": st.column_config.TextColumn("Monthly Searches", width="small"),
                        "competition": st.column_config.TextColumn("Competition", width="small")
                    },
                    hide_index=True,
                    use_container_width=True
                )

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

                # Store analysis in session state
                st.session_state.kw_messages.append({
                    "keyword": keyword_input,
                    "data": display_df.to_dict('records')
                })

        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            if st.checkbox("Show error details"):
                st.exception(e)

    # Display analysis history
    if st.session_state.kw_messages:
        st.subheader("Previous Analysis Results")
        for analysis in st.session_state.kw_messages[-5:]:  # Show last 5 analyses
            with st.expander(f"Analysis for: {analysis['keyword']}"):
                st.dataframe(
                    pd.DataFrame(analysis['data']),
                    hide_index=True,
                    use_container_width=True
                )

if __name__ == "__main__":
    run_keywordplanner_agent()
