# Import necessary libraries
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
from typing import List, Dict
import json
import pandas as pd
from pythainlp.util import normalize

def clean_thai_text(text: str) -> str:
    """Clean and normalize Thai text while preserving Thai characters"""
    if not text:
        return ""
    text = normalize(text)
    text = ' '.join(text.split())
    return text

class AdCopyGenerator:
    """Enhanced Ad Copy Generator with multi-agent approach and business intelligence"""
    
    def __init__(self, keyword_idea: str, writing_style: str = 'normal'):
        self.keyword_idea = clean_thai_text(keyword_idea)
        self.writing_style = writing_style
        
        # Set temperature based on writing style
        self.temperatures = {
            'Formal': 0.1,    # Formal style - more conservative
            'Casual': 0.3,    # Balanced approach
            'Creative': 0.7  # Creative style - more variety
        }
        
        # Initialize API keys and tools
        self.serper_api_key = st.secrets['SERPER_API_KEY']
        self.gemini_api_key = st.secrets['GEMINI_API_KEY']
        self.search_tool = SerperDevTool(api_key=self.serper_api_key)

    def _initialize_llm(self, temperature=None):
        """Initialize LLM with appropriate temperature"""
        if temperature is None:
            temperature = self.temperatures.get(self.writing_style, 0.3)
        
        return LLM(
            model="gemini/gemini-1.5-pro-latest",
            api_key=self.gemini_api_key,
            temperature=temperature
        )

    def _get_business_insights(self):
        """Get business insights from Business Analyst"""
        business_insights = {}
        if 'business_research' in st.session_state:
            business_insights = st.session_state.business_research
        return business_insights

    def _get_keyword_insights(self):
        """Get keyword insights from Keyword Planner"""
        keyword_insights = {}
        if 'top_keywords' in st.session_state:
            keyword_insights = st.session_state.top_keywords
        return keyword_insights

    def _get_website_insights(self):
        """Get website insights from Web Analyst"""
        website_insights = {}
        if 'website_analysis' in st.session_state:
            website_insights = st.session_state.website_analysis
        return website_insights

    def generate_ad_copies(self) -> Dict:
        """Main method to generate ad copies with multiple agents"""
        
        # Create Research Integration Agent
        research_integrator = Agent(
            role='Research Integration Specialist',
            goal='Collect and synthesize all available research data',
            backstory="""Expert at integrating multiple data sources and extracting 
            actionable insights for ad creation. Skilled at understanding market context 
            and audience needs.""",
            verbose=True,
            allow_delegation=True,
            tools=[self.search_tool],
            llm=self._initialize_llm(temperature=0.1)
        )

        # Create Creative Copywriter Agent
        copywriter = Agent(
            role='Creative Copywriter',
            goal='Create compelling ad copies in both Thai and English',
            backstory="""Expert bilingual copywriter skilled in crafting engaging 
            ad copy that resonates with target audiences. Specializes in creating 
            impactful headlines and descriptions.""",
            verbose=True,
            allow_delegation=False,
            tools=[],
            llm=self._initialize_llm()  # Uses style-based temperature
        )

        # Create Quality Assurance Agent
        qa_specialist = Agent(
            role='Ad Quality Specialist',
            goal='Validate ad copy against Google Ads rules and best practices',
            backstory="""Rule-based quality assurance specialist for Google Ads. 
            Validates character limits, keyword usage, and ad copy best practices 
            using predefined rules and existing research data.""",
            verbose=True,
            allow_delegation=False,
            tools=[],  # No external tools needed for rule-based validation
            llm=self._initialize_llm(temperature=0)  # Use 0 temperature for consistent rule checking
        )

        # Create Tasks
        research_task = Task(
            description=f"""
            Integrate research data for "{self.keyword_idea}":
            1. Collect business insights from Business Analyst
            2. Gather keyword data from Keyword Planner
            3. Extract website insights from Web Analyst
            4. Synthesize information for ad creation
            
            Business Data: {self._get_business_insights()}
            Keyword Data: {self._get_keyword_insights()}
            Website Data: {self._get_website_insights()}
            """,
            agent=research_integrator,
            expected_output="""
            Provide a clear list of insights in the following format:

            TARGET AUDIENCE:
            [Key characteristics of the target audience]

            KEY BENEFITS:
            1. [Benefit 1]
            2. [Benefit 2]
            3. [Benefit 3]

            COMPETITIVE ADVANTAGES:
            1. [Advantage 1]
            2. [Advantage 2]

            TONE AND STYLE RECOMMENDATIONS:
            [Specific recommendations for ad tone and style]
            """
        )

        copywriting_task = Task(
            description=f"""
            Create ad copies based on research insights:
            1. Generate 10 headlines (max 30 characters each)
            2. Create 10 descriptions (max 90 characters each)
            3. Write 10 callout texts (max 25 characters each)
            4. Create both Thai and English versions
            5. Ensure compelling and relevant content
            
            Writing Style: {self.writing_style}
            Keywords: {self.keyword_idea}
            """,
            agent=copywriter,
            expected_output="""
            THAI HEADLINES:
            1. [Headline 1 - 30 chars]
            2. [Headline 2 - 30 chars]
            ...[Continue to 10]

            THAI DESCRIPTIONS:
            1. [Description 1 - 90 chars]
            2. [Description 2 - 90 chars]
            ...[Continue to 10]

            THAI CALLOUTS:
            1. [Callout 1 - 25 chars]
            2. [Callout 2 - 25 chars]
            ...[Continue to 10]

            ENGLISH HEADLINES:
            1. [Headline 1 - 30 chars]
            ...[Continue to 10]

            ENGLISH DESCRIPTIONS:
            1. [Description 1 - 90 chars]
            ...[Continue to 10]

            ENGLISH CALLOUTS:
            1. [Callout 1 - 25 chars]
            ...[Continue to 10]
            """
        )

        qa_task = Task(
            description=f"""
            Perform rule-based validation of ad copies using the following criteria:

            CHARACTER LIMITS:
            - Headlines: Must be ≤ 30 characters
            - Descriptions: Must be ≤ 90 characters
            - Callouts: Must be ≤ 25 characters

            KEYWORD USAGE RULES:
            - Primary keyword "{self.keyword_idea}" should appear in at least:
              * 3 headlines
              * 2 descriptions
              * 1 callout
            - Check for proper keyword placement (beginning/middle/end)

            LANGUAGE VALIDATION:
            - Check for proper sentence structure
            - Verify proper capitalization
            - Ensure no excessive punctuation
            - Check for banned symbols or characters

            AD COPY RULES:
            - No repeated punctuation (!!!, ...)
            - No all-caps words (unless brand name)
            - No excessive exclamation marks
            - Must include at least one call-to-action
            - No promotional symbols ($, €, ©, ™, ®)
            - No phone numbers or URLs in headlines

            Use only the provided ad copies and research data for validation.
            """,
            agent=qa_specialist,
            expected_output="""
            VALIDATION RESULTS:

            CHARACTER LIMIT VALIDATION:
            HEADLINES:
            - Total checked: [Number]
            - Passed: [Number]
            - Failed: [List of headline numbers that exceed 30 chars]

            DESCRIPTIONS:
            - Total checked: [Number]
            - Passed: [Number]
            - Failed: [List of description numbers that exceed 90 chars]

            CALLOUTS:
            - Total checked: [Number]
            - Passed: [Number]
            - Failed: [List of callout numbers that exceed 25 chars]

            KEYWORD USAGE:
            - Headlines with keyword: [List numbers]
            - Descriptions with keyword: [List numbers]
            - Callouts with keyword: [List numbers]

            POLICY VIOLATIONS:
            1. [Specific violation and location]
            2. [Specific violation and location]
            ...

            IMPROVEMENT SUGGESTIONS:
            1. [Actionable suggestion]
            2. [Actionable suggestion]
            3. [Actionable suggestion]

            QUALITY SCORES:
            - Headline Quality: [1-10]
            - Description Quality: [1-10]
            - Callout Quality: [1-10]
            - Overall Score: [1-10]
            """
        )

        # Create and run crew
        crew = Crew(
            agents=[research_integrator, copywriter, qa_specialist],
            tasks=[research_task, copywriting_task, qa_task],
            verbose=True,
            process=Process.sequential
        )

        result = crew.kickoff()
        return self._process_results(result)

    def _process_results(self, result) -> Dict:
        """Process and structure the generation results"""
        try:
            # Initialize the data structure
            ads_data = {
                "thai": {
                    "headlines": [],
                    "descriptions": [],
                    "callouts": []
                },
                "english": {
                    "headlines": [],
                    "descriptions": [],
                    "callouts": []
                },
                "quality_scores": {},
                "recommendations": [],
                "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            # Process the raw result text
            if isinstance(result, str):
                result_text = result
            elif hasattr(result, 'raw'):
                result_text = str(result.raw)
            else:
                result_text = str(result)

            # Parse the structured text sections
            sections = result_text.split('\n\n')
            for section in sections:
                if section.startswith('THAI HEADLINES:'):
                    ads_data['thai']['headlines'] = [
                        line.split('. ')[1].strip() 
                        for line in section.split('\n')[1:] 
                        if line.strip() and '. ' in line
                    ]
                elif section.startswith('THAI DESCRIPTIONS:'):
                    ads_data['thai']['descriptions'] = [
                        line.split('. ')[1].strip() 
                        for line in section.split('\n')[1:] 
                        if line.strip() and '. ' in line
                    ]
                elif section.startswith('THAI CALLOUTS:'):
                    ads_data['thai']['callouts'] = [
                        line.split('. ')[1].strip() 
                        for line in section.split('\n')[1:] 
                        if line.strip() and '. ' in line
                    ]
                elif section.startswith('ENGLISH HEADLINES:'):
                    ads_data['english']['headlines'] = [
                        line.split('. ')[1].strip() 
                        for line in section.split('\n')[1:] 
                        if line.strip() and '. ' in line
                    ]
                elif section.startswith('ENGLISH DESCRIPTIONS:'):
                    ads_data['english']['descriptions'] = [
                        line.split('. ')[1].strip() 
                        for line in section.split('\n')[1:] 
                        if line.strip() and '. ' in line
                    ]
                elif section.startswith('ENGLISH CALLOUTS:'):
                    ads_data['english']['callouts'] = [
                        line.split('. ')[1].strip() 
                        for line in section.split('\n')[1:] 
                        if line.strip() and '. ' in line
                    ]
                elif section.startswith('QUALITY ANALYSIS:'):
                    # Extract quality scores and recommendations
                    for line in section.split('\n'):
                        if line.startswith('OVERALL QUALITY SCORE:'):
                            ads_data['quality_scores']['overall'] = line.split(':')[1].strip()
                        elif line.startswith('- '):
                            key, value = line[2:].split(': ')
                            ads_data['quality_scores'][key.lower()] = value
                        elif line.startswith('1. '):
                            ads_data['recommendations'].append(line[3:].strip())

            return ads_data
            
        except Exception as e:
            st.error(f"Error processing results: {str(e)}")
            return None

def display_ad_copies(result):
    """Display ad copies with enhanced UI and simplified layout"""
    if not result:
        st.error("No results to display")
        return

    st.header("Generated Ad Copies", divider="blue")
    
    # Thai Ads Section
    st.subheader("🇹🇭 Thai Ads")
    
    # Thai Headlines
    st.write("### Headlines (30 ตัวอักษร)")
    for i, headline in enumerate(result["thai"]["headlines"], 1):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.text_input(
                f"Headline {i}", 
                value=headline,
                key=f"thai_headline_{i}",
                help=f"Character count: {len(headline)}"
            )
        with col2:
            st.button("📋 Copy", key=f"copy_thai_headline_{i}", use_container_width=True)

    # Thai Descriptions
    st.write("### Descriptions (90 ตัวอักษร)")
    for i, desc in enumerate(result["thai"]["descriptions"], 1):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.text_area(
                f"Description {i}",
                value=desc,
                key=f"thai_desc_{i}",
                help=f"Character count: {len(desc)}",
                height=100
            )
        with col2:
            st.button("📋 Copy", key=f"copy_thai_desc_{i}", use_container_width=True)

    # Thai Callouts
    st.write("### Callouts (25 ตัวอักษร)")
    callout_cols = st.columns(2)
    for i, callout in enumerate(result["thai"]["callouts"], 1):
        with callout_cols[i % 2]:
            st.text_input(
                f"Callout {i}",
                value=callout,
                key=f"thai_callout_{i}",
                help=f"Character count: {len(callout)}"
            )
            st.button("📋 Copy", key=f"copy_thai_callout_{i}", use_container_width=True)
    
    st.divider()

    # English Ads Section
    st.subheader("🇬🇧 English Ads")
    
    # English Headlines
    st.write("### Headlines (30 characters)")
    for i, headline in enumerate(result["english"]["headlines"], 1):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.text_input(
                f"Headline {i}",
                value=headline,
                key=f"eng_headline_{i}",
                help=f"Character count: {len(headline)}"
            )
        with col2:
            st.button("📋 Copy", key=f"copy_eng_headline_{i}", use_container_width=True)

    # English Descriptions
    st.write("### Descriptions (90 characters)")
    for i, desc in enumerate(result["english"]["descriptions"], 1):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.text_area(
                f"Description {i}",
                value=desc,
                key=f"eng_desc_{i}",
                help=f"Character count: {len(desc)}",
                height=100
            )
        with col2:
            st.button("📋 Copy", key=f"copy_eng_desc_{i}", use_container_width=True)

    # English Callouts
    st.write("### Callouts (25 characters)")
    callout_cols = st.columns(2)
    for i, callout in enumerate(result["english"]["callouts"], 1):
        with callout_cols[i % 2]:
            st.text_input(
                f"Callout {i}",
                value=callout,
                key=f"eng_callout_{i}",
                help=f"Character count: {len(callout)}"
            )
            st.button("📋 Copy", key=f"copy_eng_callout_{i}", use_container_width=True)

    st.divider()

    # Quality Analysis Section
    st.subheader("📊 Quality Analysis")
    
    # Quality Scores
    if "quality_scores" in result:
        st.write("### Quality Scores")
        score_cols = st.columns(4)
        for idx, (metric, score) in enumerate(result["quality_scores"].items()):
            with score_cols[idx % 4]:
                st.metric(
                    label=metric.title(),
                    value=score,
                    delta=None,
                    help="Score out of 10"
                )
    
    # Recommendations
    if "recommendations" in result:
        st.write("### Recommendations")
        for idx, rec in enumerate(result["recommendations"], 1):
            st.info(f"{idx}. {rec}")

    # Add timestamp
    st.caption(f"Generated at: {result.get('generated_at', 'N/A')}")

def run_adcopy():
    """Main function to run the Ad Copy Generator"""
    
    st.title("Ad Copy Generator")
    st.markdown("""
    Generate optimized ad copies for your campaigns.
    Supports both Thai and English languages.
    """)

    with st.form("ad_copy_form"):
        keyword_idea = st.text_input(
            "Keyword Idea",
            help="Enter your main keyword or topic"
        )

        writing_style = st.select_slider(
            "Writing Style",
            options=["Formal", "Casual", "Creative"],
            value="Casual",
            help="Select your preferred writing style"
        )

        generate_button = st.form_submit_button(
            "Generate Ad Copies",
            type="primary",
            use_container_width=True
        )

    if generate_button and keyword_idea:
        with st.spinner('Generating ad copies... This may take a few minutes.'):
            try:
                generator = AdCopyGenerator(
                    keyword_idea=keyword_idea,
                    writing_style=writing_style
                )
                
                result = generator.generate_ad_copies()
                display_ad_copies(result)
                
            except Exception as e:
                st.error(f"An error occurred during generation: {str(e)}")

if __name__ == "__main__":
    run_adcopy()
