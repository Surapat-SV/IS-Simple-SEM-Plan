# Import necessary libraries
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union
import httpx
from bs4 import BeautifulSoup
import json
import pandas as pd
import re
from datetime import datetime

# Define Pydantic models for structured data
class KeywordAnalysis(BaseModel):
    keyword: str
    score: float

class KeywordComparison(BaseModel):
    common_keywords: List[str]
    aquapro_unique_keywords: List[str]
    goprothai_unique_keywords: List[str]
    opportunities: List[str]

class AnalysisOutput(BaseModel):
    top_keywords: List[KeywordAnalysis]
    keyword_comparison: KeywordComparison
    similarity_score: float

class CustomWebScraper(BaseTool):
    name: str = Field(default="Custom Web Scraper")
    description: str = Field(default="Scrapes website content using BeautifulSoup")

    class InputSchema(BaseModel):
        website_url: str = Field(..., description="URL of the website to scrape")

    def _run(self, website_url: str) -> Dict[str, Any]:
        try:
            # Configure headers
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            # Make request
            with httpx.Client() as client:
                response = client.get(website_url, headers=headers)
                response.raise_for_status()

            # Parse with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract metadata
            title = soup.find('title')
            title_text = title.text.strip() if title else ""

            description = soup.find('meta', {'name': 'description'})
            description_text = description.get('content', '').strip() if description else ""

            # Extract main content
            # Remove unwanted elements
            for tag in soup(['script', 'style', 'iframe', 'nav', 'footer']):
                tag.decompose()

            # Get text content
            text = soup.get_text(separator=' ', strip=True)

            # Clean the text
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            cleaned_text = ' '.join(lines)

            return {
                "url": website_url,
                "metadata": {
                    "title": title_text,
                    "description": description_text
                },
                "content": cleaned_text,
                "html": str(soup)
            }

        except Exception as e:
            return {
                "url": website_url,
                "error": str(e),
                "metadata": {"title": "", "description": ""},
                "content": "",
                "html": ""
            }

class WebsiteContentAnalyzer:
    def __init__(self, website_url: str, competitor_urls: Optional[List[str]] = None, top_n: int = 10):
        self.website_url = website_url
        self.competitor_urls = competitor_urls if competitor_urls else []
        self.top_n = top_n
        self.serper_api_key = st.secrets['SERPER_API_KEY']
        self.gemini_api_key = st.secrets['GEMINI_API_KEY']
        
        # Initialize custom tools
        self.custom_scraper = CustomWebScraper()
        self.search_tool = SerperDevTool()
        
    def _initialize_llm(self, temperature: float = 0) -> LLM:
        return LLM(
            model="gemini/gemini-1.5-pro-latest",
            api_key=self.gemini_api_key,
            temperature=temperature
        )

    def _create_agents(self) -> tuple[Agent, Agent]:
        # Data Collection Agent
        scraper = Agent(
            role='Web Scraper',
            goal='Collect website content and metadata efficiently',
            backstory="""Expert at extracting website content and metadata. 
            Focuses on getting clean, relevant data for analysis.""",
            verbose=True,
            allow_delegation=False,
            tools=[self.custom_scraper],
            llm=self._initialize_llm()
        )

        # NLP Analysis Agent
        analyzer = Agent(
            role='NLP Analyst',
            goal='Analyze website content using advanced NLP techniques',
            backstory="""Specialist in Thai NLP and semantic analysis. 
            Expert at extracting insights from website content.""",
            verbose=True,
            allow_delegation=False,
            tools=[],
            llm=self._initialize_llm()
        )

        return scraper, analyzer

    def _create_tasks(self, scraper: Agent, analyzer: Agent) -> List[Task]:
        # Content Collection Task
        scraping_task = Task(
            description=f"""
            Analyze these websites using the custom web scraper:
            1. Main website: {self.website_url}
            2. Competitor website(s): {', '.join(self.competitor_urls)}
            
            Extract and return:
            1. Main content
            2. Metadata (title, description)
            3. Clean and structured text
            
            Format the results as:
            {{
                "main_site": {{
                    "url": "...",
                    "content": "...",
                    "metadata": {{
                        "title": "...",
                        "description": "..."
                    }}
                }},
                "competitors": [
                    {{
                        "url": "...",
                        "content": "...",
                        "metadata": {{
                            "title": "...",
                            "description": "..."
                        }}
                    }}
                ]
            }}
            """,
            agent=scraper,
            expected_output="A JSON object containing scraped content and metadata for main site and competitors"
        )

        # Content Analysis Task
        analysis_task = Task(
            description=f"""
            Analyze the website content and ensure the output matches this exact structure:
            {{
                "top_keywords": [
                    {{"keyword": "word1", "score": 0.8}},
                    {{"keyword": "word2", "score": 0.6}}
                ],
                "keyword_comparison": {{
                    "common_keywords": ["keyword1", "keyword2"],
                    "aquapro_unique_keywords": ["unique1", "unique2"],
                    "goprothai_unique_keywords": ["comp1", "comp2"],
                    "opportunities": ["opportunity1", "opportunity2"]
                }},
                "similarity_score": 0.75
            }}
            
            Important rules:
            1. All numeric values must be valid float numbers between 0 and 1
            2. The similarity_score must always be a float
            3. All lists must be non-null arrays
            4. All keywords must be non-empty strings
            """,
            agent=analyzer,
            expected_output="A JSON object containing keyword analysis, comparison results, and content insights"
        )

        return [scraping_task, analysis_task]

    def _extract_research_data(self, crew_output: Any) -> Dict[str, Any]:
        """Extract structured data from crew output with proper error handling"""
        try:
            # Get the text content from CrewOutput
            output_text = str(crew_output.result if hasattr(crew_output, 'result') else crew_output)
            
            # Look for JSON content between triple backticks
            json_match = re.search(r'```json\s*(.*?)\s*```', output_text, re.DOTALL)
            if json_match:
                # Parse the JSON content
                data = json.loads(json_match.group(1))
                
                # Validate the data structure
                try:
                    analysis_output = AnalysisOutput(**data)
                    data = analysis_output.dict()
                except Exception as e:
                    st.warning(f"Data validation error: {str(e)}")
                    # Provide default values if validation fails
                    data = {
                        "top_keywords": [],
                        "keyword_comparison": {
                            "common_keywords": [],
                            "aquapro_unique_keywords": [],
                            "goprothai_unique_keywords": [],
                            "opportunities": []
                        },
                        "similarity_score": 0.0
                    }
                
                # Structure the data with safe default values
                structured_data = {
                    "keywords": {
                        "main_site": data.get("top_keywords", []),
                        "competitors": []
                    },
                    "insights": {
                        "similarity_score": float(data.get("similarity_score", 0.0)),
                        "keyword_comparison": data.get("keyword_comparison", {})
                    },
                    "recommendations": {
                        "common_keywords": data.get("keyword_comparison", {}).get("common_keywords", []),
                        "unique_keywords": {
                            "main_site": data.get("keyword_comparison", {}).get("aquapro_unique_keywords", []),
                            "competitor": data.get("keyword_comparison", {}).get("goprothai_unique_keywords", [])
                        },
                        "opportunities": data.get("keyword_comparison", {}).get("opportunities", [])
                    }
                }
                
                return structured_data
            else:
                return {
                    "keywords": {"main_site": [], "competitors": []},
                    "insights": {"similarity_score": 0.0},
                    "recommendations": {"opportunities": []}
                }
                
        except Exception as e:
            st.warning(f"Error processing analysis data: {str(e)}")
            return {
                "error": str(e),
                "raw_output": output_text if 'output_text' in locals() else "No output available",
                "keywords": {"main_site": [], "competitors": []},
                "insights": {"similarity_score": 0.0},
                "recommendations": {"opportunities": []}
            }

    def analyze_website(self) -> Optional[Dict[str, Any]]:
        try:
            # Initialize agents and tasks
            scraper, analyzer = self._create_agents()
            tasks = self._create_tasks(scraper, analyzer)

            # Create and run crew
            crew = Crew(
                agents=[scraper, analyzer],
                tasks=tasks,
                verbose=True,
                process=Process.sequential
            )

            # Run analysis
            result = crew.kickoff()
            
            # Process analysis results
            processed_data = self._extract_research_data(result)
            
            # Store raw output for debugging
            st.session_state['raw_agent_output'] = str(result)
            
            # Format final output
            final_data = {
                "website": self.website_url,
                "analysis_results": processed_data,
                "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            return final_data
                    
        except Exception as e:
            st.error(f"Error in analysis: {str(e)}")
            if 'raw_agent_output' in st.session_state:
                st.expander("Debug: Raw Agent Output").code(st.session_state['raw_agent_output'])
            return None

def run_webanalyst(website_url: Optional[str] = None) -> None:
    st.title("Website Keyword Analyzer")
    st.caption("Analyze and compare website content using NLP")

    # Use provided website_url or get from session state
    if not website_url:
        website_url = st.session_state.get('website_url', '')

    with st.form("website_analysis_form"):
        if not website_url:
            website_url = st.text_input(
                "Main Website URL",
                help="Enter the full URL starting with http:// or https://"
            )
        else:
            st.text_input(
                "Main Website URL",
                value=website_url,
                disabled=True
            )
        
        competitors = st.text_area(
            "Competitor URLs",
            help="Enter competitor URLs, one per line"
        )
        
        analyze_button = st.form_submit_button(
            "Analyze Websites",
            type="primary",
            use_container_width=True
        )

    if analyze_button and website_url:
        with st.spinner('Analyzing websites... This may take a few minutes.'):
            try:
                competitor_urls = [
                    url.strip() 
                    for url in competitors.split('\n') 
                    if url.strip()
                ]
                
                analyzer = WebsiteContentAnalyzer(
                    website_url=website_url,
                    competitor_urls=competitor_urls
                )
                
                # Run analysis and process results
                result = analyzer.analyze_website()
                
                if result and 'analysis_results' in result:
                    analysis_data = result['analysis_results']
                    st.session_state['analysis_results'] = analysis_data

                    # Display keyword analysis
                    if 'keywords' in analysis_data:
                        keywords = analysis_data['keywords']
                        
                        # Main site keywords
                        if keywords['main_site']:
                            st.header("Our Site Keywords")
                            main_df = pd.DataFrame(keywords['main_site'])
                            st.bar_chart(data=main_df, x='keyword', y='score')
                        
                        # Competitor keywords
                        if keywords['competitors']:
                            st.header("Competitor Keywords")
                            comp_df = pd.DataFrame(keywords['competitors'])
                            st.bar_chart(data=comp_df, x='keyword', y='score')

                    # Display insights
                    if 'insights' in analysis_data:
                        st.header("Website Insights")
                        insights = analysis_data['insights']
                        
                        # Display similarity score with safe type conversion
                        if 'similarity_score' in insights:
                            similarity_score = float(insights.get('similarity_score', 0.0))
                            st.metric(
                                "Site Keyword Similarity Score", 
                                f"{similarity_score * 100:.1f}%"
                            )
                        # Display keyword gaps
                        if 'keyword_comparison' in insights:
                            st.subheader("Keyword Analysis")
                            comp = insights['keyword_comparison']
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("Common Keywords:")
                                for kw in comp.get('common_keywords', []):
                                    st.write(f"- {kw}")
                            
                            with col2:
                                st.write("Unique Keywords:")
                                st.write("Your site:")
                                for kw in comp.get('aquapro_unique_keywords', []):
                                    st.write(f"- {kw}")
                                st.write("Competitor site:")
                                for kw in comp.get('goprothai_unique_keywords', []):
                                    st.write(f"- {kw}")

                    # Display recommendations
                    if 'recommendations' in analysis_data:
                        st.header("Recommendations")
                        recs = analysis_data['recommendations']
                        
                        if 'opportunities' in recs:
                            st.subheader("Opportunities")
                            for opp in recs['opportunities']:
                                st.write(f"- {opp}")

                    # Show raw data in expander for debugging
                    with st.expander("Raw Analysis Data", expanded=False):
                        st.json(result)
                            
                else:
                    st.error("Analysis failed to produce results")
                    if result:
                        st.json(result)
                        
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.write("Error details:", str(e))
                if 'result' in locals():
                    with st.expander("Debug Data", expanded=True):
                        st.write(result)

if __name__ == "__main__":
    run_webanalyst()
