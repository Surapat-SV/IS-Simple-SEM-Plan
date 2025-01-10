# Import necessary libraries
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from streamlit_option_menu import option_menu
import os
import json
import openai
from google.oauth2 import service_account
from crewai_tools import SerperDevTool

# Import views
from view.businessanalyst import run_business_analyst_chatbot
from view.webanalyst import run_webanalyst
from view.keywordplanner import run_keywordplanner_agent
from view.adcopy import run_adcopy

# Initialize session state if needed
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.history = []

def initialize_credentials():
    """Initialize all required API keys and credentials."""
    try:
        # Load OpenAI API key
        openai.api_key = st.secrets["OPENAI_API_KEY"]

        # Load Serper API key
        serper_api_key = st.secrets["SERPER_API_KEY"]
        serper_tool = SerperDevTool(api_key=serper_api_key)

        # Load Google credentials
        try:
            # If using JSON content directly
            service_account_json = st.secrets["general"].get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
            if service_account_json:
                google_credentials_info = json.loads(service_account_json)
                google_credentials = service_account.Credentials.from_service_account_info(google_credentials_info)
            else:
                # If using a file path
                google_credentials_path = st.secrets["general"].get("GOOGLE_APPLICATION_CREDENTIALS")
                if google_credentials_path:
                    google_credentials = service_account.Credentials.from_service_account_file(google_credentials_path)
                else:
                    raise KeyError("Missing both GOOGLE_APPLICATION_CREDENTIALS_JSON and GOOGLE_APPLICATION_CREDENTIALS.")
            
            st.success("Google credentials initialized successfully!")
        except json.JSONDecodeError as e:
            st.error("Error decoding GOOGLE_APPLICATION_CREDENTIALS_JSON. Please check your secrets.toml.")
            st.error(f"Details: {str(e)}")
        except Exception as e:
            st.error(f"An error occurred while initializing Google credentials: {str(e)}")


        # Load Gemini API key (if applicable)
        gemini_api_key = st.secrets.get("GEMINI_API_KEY", "")
        if not gemini_api_key:
            st.warning("Gemini API key is not set in secrets. Some features may not work.")

        st.success("All credentials initialized successfully!")
        return {
            "serper_tool": serper_tool,
            "google_credentials": google_credentials,
            "gemini_api_key": gemini_api_key
        }

    except json.JSONDecodeError as e:
        st.error("Error decoding GOOGLE_APPLICATION_CREDENTIALS JSON. Please check your secrets.toml.")
        st.error(f"Details: {str(e)}")
        return None

    except KeyError as e:
        st.error(f"Missing required key in secrets: {str(e)}")
        return None

# Configure Streamlit page
st.set_page_config(
    page_title="SEM Planner - AI Powered App",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://yourwebsite.com/help',
        'Report a bug': "https://yourwebsite.com/bug",
        'About': """
        # SEM Planner AI App
        AI-powered SEM planning and optimization tool.
        Version 1.0.0
        """
    }
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea {
        background-color: grey;
    }
    .reportview-container .main .block-container {
        max-width: 95%;
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.title("SEM Planner")
    st.image("https://via.placeholder.com/150", width=150)  # Replace with your logo
    
    st.info("""
        Plan and Optimize SEM Campaigns:
        - 📋 Define Target Audience
        - 🌐 Analyze Websites and Keywords
        - 🔑 Optimize Keyword Strategies
        - ✍️ Generate Ad Copies
    """)
    
    # Navigation menu using streamlit_option_menu
    selected = option_menu(
        menu_title="Navigation",
        options=[
            "Business Analyst",
            "Web Analyst", 
            "Keyword Planner",
            "Ad Copywriter"
        ],
        icons=[
            'briefcase',
            'globe',
            'key',
            'pencil'
        ],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#262730"},
            "icon": {"color": "orange", "font-size": "25px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "#0083B8"},
        },
    )
    
    # Add footer to sidebar
    st.sidebar.markdown("---")
    st.sidebar.caption("© 2024 SEM Planner. All rights reserved.")

# Main app routing
try:
    if selected == "Business Analyst":
        run_business_analyst_chatbot()
    elif selected == "Web Analyst":
        run_webanalyst()
    elif selected == "Keyword Planner":
        run_keywordplanner_agent()
    elif selected == "Ad Copywriter":
        run_adcopy()

except Exception as e:
    st.error(f"""
        An error has occurred: {str(e)}
        
        Please try again or contact support if the problem persists.
    """)
    
    if st.checkbox("Show error details"):
        st.exception(e)

# Add footer to main content
st.markdown("---")
cols = st.columns([2, 1, 1])
with cols[0]:
    st.caption("Made with ❤️ by Your Company Name")
with cols[1]:
    st.caption("[Documentation](https://yourwebsite.com/docs)")
with cols[2]:
    st.caption("[Support](https://yourwebsite.com/support)")

# Initialize environment variables and configuration
def init_environment():
    """Initialize environment variables and check API keys"""
    required_env_vars = [
        'OPENAI_API_KEY',
        'SERPER_API_KEY',
        'GOOGLE_APPLICATION_CREDENTIALS'
    ]
    
    missing_vars = []
    for var in required_env_vars:
        if var not in st.secrets:
            missing_vars.append(var)
    
    if missing_vars:
        st.warning(f"""
            Missing required environment variables: {', '.join(missing_vars)}
            Please set these variables in your secrets.toml file.
        """)
        return False
    
    return True

# Run environment check
if not init_environment():
    st.error("""
        Application not properly configured. 
        Please check your environment variables and try again.
    """)
    st.stop()

# Add application version info
if st.sidebar.checkbox("Show Version Info"):
    st.sidebar.info("""
        Version: 1.0.0
        Last Updated: 2024-01-09
        Python: 3.9+
        Streamlit: 1.24+
    """)

# Add usage tracking (optional)
if 'page_views' not in st.session_state:
    st.session_state.page_views = 0
st.session_state.page_views += 1

# Debugging mode toggle (if needed)
if st.sidebar.checkbox("Debug Mode", key="debug_mode"):
    st.sidebar.write(f"Session State: {dict(st.session_state)}")
    st.sidebar.write(f"Current Page: {selected}")
    st.sidebar.write(f"Page Views: {st.session_state.page_views}")
