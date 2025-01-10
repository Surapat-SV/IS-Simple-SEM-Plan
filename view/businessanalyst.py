# Ensure compatibility for SQLite in certain environments
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from dotenv import load_dotenv
import json
import pandas as pd

# Load environment variables
load_dotenv()

def run_businessanalyst():
    """Main function to run the Business Analyst tool and store results"""
    
    st.title("Business Research Tool")
    st.markdown("Collect and analyze business information for further processing.")

    # Input form
    with st.form("business_research_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            business_name = st.text_input("Business Name")
            product_service = st.text_input("Product or Service")

        with col2:
            website_url = st.text_input("Website URL")
            industry = st.selectbox(
                "Industry",
                ["Technology", "E-commerce", "Healthcare", "Finance", 
                 "Education", "Real Estate", "Manufacturing", "Other"]
            )

        target_audience_input = st.text_area(
            "Target Audience Description",
            height=100
        )

        submitted = st.form_submit_button("Research", type="primary", use_container_width=True)

    if submitted and all([business_name, product_service, target_audience_input, website_url]):
        with st.spinner('Collecting business information...'):
            try:
                # Initialize researcher
                researcher = BusinessResearcher(
                    business_name=business_name,
                    product_service=product_service,
                    target_audience=target_audience_input,
                    website_url=website_url,
                    industry=industry
                )
                
                # Collect business data
                research_data = researcher.collect_data()
                
                if research_data:
                    # Store the JSON data in session state for later use
                    st.session_state.business_research = research_data
                    
                    # Display the collected information
                    display_research_results(research_data)
                    
                    # Show raw JSON for debugging/verification
                    with st.expander("Raw Research Data (JSON)", expanded=False):
                        st.code(json.dumps(research_data, indent=2), language='json')

            except Exception as e:
                st.error(f"Research error: {str(e)}")
    elif submitted:
        st.warning("Please fill in all required fields.")

class BusinessResearcher:
    """Class to handle business research and data collection"""
    
    def __init__(self, business_name, product_service, target_audience, website_url, industry):
        self.business_name = business_name
        self.product_service = product_service
        self.target_audience = target_audience
        self.website_url = website_url
        self.industry = industry
        self.serper_api_key = st.secrets['SERPER_API_KEY']
        self.gemini_api_key = st.secrets['GEMINI_API_KEY']
        self.llm = LLM(
            model="gemini/gemini-1.5-pro-latest",
            api_key=self.gemini_api_key,
            temperature=0
        )
        self.search_tool = SerperDevTool(api_key=self.serper_api_key)
        self.scrape_tool = ScrapeWebsiteTool()

    def _create_researcher_agent(self):
        """Create research agent"""
        return Agent(
            role="Business Researcher",
            goal=f"Research and collect comprehensive data about {self.business_name}",
            backstory="Expert in business research and data collection",
            tools=[self.search_tool, self.scrape_tool],
            allow_delegation=False,
            llm=self.llm
        )

    def _create_research_task(self, agent):
        """Create research task"""
        return Task(
            description=f"""
            Research and collect data for {self.business_name}. 
            Focus on gathering factual, verifiable information.

            Business Details:
            - Name: {self.business_name}
            - Website: {self.website_url}
            - Product/Service: {self.product_service}
            - Target Audience: {self.target_audience}
            - Industry: {self.industry}

            Collect and structure the following information:
            1. Basic company information
            2. Product/service details
            3. Target market information
            4. Basic competitive position
            5. Notable online presence details

            Format all information in clear, concise text suitable for future analysis.
            """,
            agent=agent,
            expected_output="Comprehensive business data in structured format"
        )

    def collect_data(self):
        """Collect and structure business research data"""
        try:
            researcher = self._create_researcher_agent()
            research_task = self._create_research_task(researcher)
            
            crew = Crew(
                agents=[researcher],
                tasks=[research_task],
                verbose=True,
                process=Process.sequential
            )

            # Get research results
            result = crew.kickoff()
            
            # Structure the data
            research_data = {
                "business_info": {
                    "name": self.business_name,
                    "website": self.website_url,
                    "industry": self.industry,
                    "product_service": self.product_service,
                    "target_audience": self.target_audience
                },
                "research_data": self._extract_research_data(result),
                "collected_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return research_data
            
        except Exception as e:
            st.error(f"Data collection error: {str(e)}")
            return None

    def _extract_research_data(self, result):
        """Extract and structure research data from result"""
        try:
            if hasattr(result, 'result'):
                return str(result.result)
            elif hasattr(result, 'raw'):
                return str(result.raw)
            return str(result)
        except Exception as e:
            st.error(f"Data extraction error: {str(e)}")
            return "Error extracting research data"

def display_research_results(data):
    """Display collected research data"""
    st.header("Collected Business Information")
    
    # Display basic business info
    st.subheader("Business Details")
    business_info = data["business_info"]
    st.write(f"**Name:** {business_info['name']}")
    st.write(f"**Industry:** {business_info['industry']}")
    st.write(f"**Website:** {business_info['website']}")
    
    # Display research findings
    st.subheader("Research Findings")
    st.write(data["research_data"])
    
    # Show collection timestamp
    st.caption(f"Information collected at: {data['collected_at']}")
    
    # Success message with next steps
    st.success("""
        Research data has been collected and stored for further analysis.
        This information will be available to subsequent analysis tools.
    """)

if __name__ == "__main__":
    run_businessanalyst()
