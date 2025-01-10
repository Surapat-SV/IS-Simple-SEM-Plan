__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from dotenv import load_dotenv
import pandas as pd
from pythainlp.util import normalize
import plotly.express as px
from typing import List, Dict
from textwrap import dedent
from datetime import date
from google.cloud import bigquery
from google.oauth2 import service_account
import json

load_dotenv()

def clean_thai_text(text: str) -> str:
    """Clean and normalize Thai text"""
    if not text:
        return ""
    text = normalize(text)
    text = ' '.join(text.split())
    return text

class BigQueryKeywordTool(BaseTool):
    name: str = "BigQuery Keyword Data Tool"
    description: str = "Fetches keyword data from BigQuery database with monthly searches and competition data"
    
    def _clean_string_for_query(self, s: str) -> str:
        """Clean and escape string for SQL query"""
        if not s:
            return ""
        # Remove any control characters
        s = ''.join(char for char in s if ord(char) >= 32)
        # Escape single quotes
        s = s.replace("'", "\\'")
        return s

    def _run(self, keyword: str) -> str:
        try:
            # Parse the input keyword if it's a JSON string
            if keyword.startswith('{'):
                try:
                    keyword_data = json.loads(keyword)
                    keyword = keyword_data.get('keyword', '')
                except json.JSONDecodeError:
                    keyword = keyword
            
            # Clean the keyword for the query
            cleaned_keyword = self._clean_string_for_query(keyword)
            
            # Get credentials
            service_account_json = st.secrets["general"]["GOOGLE_APPLICATION_CREDENTIALS_JSON"]
            service_account_info = json.loads(service_account_json)
            credentials = service_account.Credentials.from_service_account_info(service_account_info)
            client = bigquery.Client(credentials=credentials, project=service_account_info["project_id"])
            
            # Use a parameterized query
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
                    bigquery.ScalarQueryParameter("keyword_pattern", "STRING", f"%{cleaned_keyword}%"),
                ]
            )
            
            # Execute query and get results
            df = client.query(query, job_config=job_config).result().to_dataframe()
            
            # Convert competition values to Thai
            competition_map = {
                'HIGH': 'สูง',
                'MEDIUM': 'ปานกลาง',
                'LOW': 'ต่ำ'
            }
            if 'competition' in df.columns:
                df['competition'] = df['competition'].map(competition_map).fillna('ไม่ระบุ')
            
            # Handle NULL values
            df = df.fillna({
                'avg_monthly_searches': 0,
                'keyword': 'ไม่ระบุ'
            })
            
            # Return DataFrame as a string representation that can be evaluated later
            return df.to_string()
            
        except Exception as e:
            return f"Error: {str(e)}"

class KeywordPlannerTasks:
    def keyword_discovery_task(self, agent, query_input):
        return Task(
            description=dedent(f"""
                Perform advanced keyword discovery using BigQuery and search tools.
                Query databases for keywords related to the input topic and identify
                high-potential keywords based on relevance, search volume, and competition level.
                Analysis Steps:
                1. Use BigQuery to fetch comprehensive keyword data for {query_input}
                2. Analyze search volume trends and competition metrics
                3. Filter and rank keywords based on multiple performance indicators
                4. Return results in original Thai language with proper formatting
                5. Include detailed metrics for each keyword
                
                User Input: {query_input}
            """),
            expected_output="A dictionary with detailed keyword data and comprehensive metrics",
            agent=agent
        )

    def keyword_competitor_analysis_task(self, agent, query_input):
        return Task(
            description=dedent(f"""
                Perform in-depth competitor analysis for keywords related to "{query_input}":
                Analysis Steps:
                1. Query BigQuery for detailed competitor metrics
                2. Assess competition level and difficulty for each keyword
                3. Analyze ranking patterns and competitor strategies
                4. Map the competitive landscape with specific insights
                5. Identify strategic keyword gaps and opportunities
                6. Provide actionable recommendations
                
                Maintain all Thai keywords in original form with proper formatting.
            """),
            expected_output="A dictionary with comprehensive competition data and strategic insights",
            agent=agent
        )

class KeywordAnalyzer:
    def __init__(self, keyword_idea: str):
        self.keyword_idea = clean_thai_text(keyword_idea)
        self.gemini_api_key = st.secrets['GEMINI_API_KEY']
        self.bigquery_keyword_tool = BigQueryKeywordTool()
        self.tasks = KeywordPlannerTasks()

    def _initialize_llm(self, temperature=0.1):
        return LLM(
            model="gemini/gemini-1.5-pro-latest",
            api_key=self.gemini_api_key,
            temperature=temperature
        )

    def analyze_keywords(self) -> Dict:
        # Initialize agents with specific roles and tools
        keyword_researcher = Agent(
            role='Keyword Research Specialist',
            goal=f'Research and analyze keywords related to "{self.keyword_idea}"',
            backstory="""Expert in keyword research and analysis, specializing in Thai language SEO. 
            Skilled at identifying valuable keywords and search patterns in Thai markets.""",
            verbose=True,
            allow_delegation=False,
            tools=[self.bigquery_keyword_tool],
            llm=self._initialize_llm()
        )

        competition_analyst = Agent(
            role='Competition Analyst',
            goal=f'Analyze competition and difficulty for keywords related to "{self.keyword_idea}"',
            backstory="""Specialist in competitive analysis for Thai market keywords. 
            Expert at assessing keyword difficulty and competition levels.""",
            verbose=True,
            allow_delegation=False,
            tools=[self.bigquery_keyword_tool],
            llm=self._initialize_llm()
        )

        # Create tasks using the task planner
        research_task = self.tasks.keyword_discovery_task(keyword_researcher, self.keyword_idea)
        competition_task = self.tasks.keyword_competitor_analysis_task(competition_analyst, self.keyword_idea)

        # Set up and execute the crew
        crew = Crew(
            agents=[keyword_researcher, competition_analyst],
            tasks=[research_task, competition_task],
            verbose=True,
            process=Process.sequential
        )

        result = crew.kickoff()
        return self._process_results(result)

    def _process_results(self, result) -> Dict:
        try:
            # Handle string result by parsing it as JSON
            if isinstance(result, str):
                try:
                    parsed_result = json.loads(result)
                except json.JSONDecodeError:
                    return {
                        "keyword_analysis": {"keywords": []},
                        "competition_analysis": {
                            "insights": [result]
                        },
                        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
            else:
                parsed_result = result.raw if hasattr(result, 'raw') else result

            if isinstance(parsed_result, dict):
                return {
                    "keyword_analysis": parsed_result.get("keyword_analysis", {"keywords": []}),
                    "competition_analysis": parsed_result.get("competition_analysis", {"insights": []}),
                    "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            else:
                return {
                    "keyword_analysis": {"keywords": []},
                    "competition_analysis": {
                        "insights": ["Analysis completed but returned unexpected format."]
                    },
                    "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                }
        except Exception as e:
            st.error(f"Error processing results: {str(e)}")
            return {
                "keyword_analysis": {"keywords": []},
                "competition_analysis": {
                    "insights": [f"Error processing results: {str(e)}"]
                },
                "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }

def display_keyword_results(result):
    if not result:
        st.error("No results to display")
        return

    try:
        # Display Keyword Research Results
        st.header("Keyword Research Results")
        
        # Try to convert the result string back to a DataFrame
        try:
            # If result is a string representation of a DataFrame
            if isinstance(result, str) and "DataFrame" in result:
                # Convert string back to DataFrame using pandas read_csv with StringIO
                from io import StringIO
                df = pd.read_csv(StringIO(result), sep='\s+')
            else:
                # If result is in another format, try to parse it
                data = result.get("keyword_analysis", {}).get("keywords", [])
                if isinstance(data, str):
                    data = json.loads(data)
                df = pd.DataFrame(data)
        except Exception as e:
            st.error(f"Could not parse results: {str(e)}")
            return

        if len(df) > 0:
            # Format the DataFrame
            if "avg_monthly_searches" in df.columns:
                df["avg_monthly_searches"] = df["avg_monthly_searches"].apply(
                    lambda x: "{:,}".format(int(float(x))) if pd.notnull(x) and str(x).replace(".", "").isdigit() else x
                )
            
            # Display the data in a Streamlit table
            st.write("### Keyword Data")
            st.dataframe(
                df,
                column_config={
                    "keyword": st.column_config.TextColumn("คีย์เวิร์ด", width="medium"),
                    "avg_monthly_searches": st.column_config.TextColumn("ปริมาณการค้นหาต่อเดือน", width="small"),
                    "competition": st.column_config.TextColumn("การแข่งขัน", width="small")
                },
                hide_index=True,
                use_container_width=True
            )

            # Create visualization
            if len(df) > 0 and "avg_monthly_searches" in df.columns:
                st.write("### Top Keywords Visualization")
                try:
                    # Convert formatted numbers back to integers for plotting
                    plot_df = df.copy()
                    plot_df["avg_monthly_searches"] = plot_df["avg_monthly_searches"].apply(
                        lambda x: int(str(x).replace(",", "")) if isinstance(x, str) and str(x).replace(",", "").isdigit() else 0
                    )
                    
                    # Create bar chart
                    fig = px.bar(
                        plot_df.head(10),
                        x="keyword",
                        y="avg_monthly_searches",
                        title="Top 10 Keywords by Monthly Searches",
                        labels={"keyword": "คีย์เวิร์ด", "avg_monthly_searches": "ปริมาณการค้นหาต่อเดือน"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not create visualization: {str(e)}")
        else:
            st.info("No keyword data available.")

        # Competition Analysis
        st.header("Competition Analysis")
        competition_data = result.get("competition_analysis", {})
        insights = competition_data.get("insights", [])
        
        if insights:
            st.write("### Key Insights")
            for insight in insights:
                if insight:
                    st.markdown(f"• {insight}")
        else:
            st.info("No competition insights available.")

        # Analysis Timestamp
        st.caption(f"Analysis generated at: {result.get('generated_at', 'N/A')}")
        
    except Exception as e:
        st.error(f"Error displaying results: {str(e)}")
        if st.checkbox("Show debugging information"):
            st.code(result)

def run_keywordplanner_agent():
    st.title("Keyword Planner")
    st.markdown("""
    This tool helps analyze keywords and develop content strategies.
    Supports Thai language keywords and analysis.
    """)

    with st.form("keyword_analysis_form"):
        keyword_input = st.text_input(
            "Enter Keyword or Topic",
            help="Enter your main keyword or topic in Thai or English"
        )

        analyze_button = st.form_submit_button(
            "Analyze Keywords",
            type="primary",
            use_container_width=True
        )

    if analyze_button and keyword_input:
        with st.spinner('Analyzing keywords... This may take a few minutes.'):
            try:
                analyzer = KeywordAnalyzer(keyword_input)
                result = analyzer.analyze_keywords()
                display_keyword_results(result)
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")

if __name__ == "__main__":
    run_keywordplanner_agent()
