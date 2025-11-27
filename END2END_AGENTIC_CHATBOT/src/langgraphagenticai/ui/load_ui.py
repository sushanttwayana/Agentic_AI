import streamlit as st
import os

from src.langgraphagenticai.ui.uiconfigfile import Config

class LoadStreamlitUI:
    def __init__(self):
        self.config = Config()
        self.user_controls = {}

    def load_streamlit_ui(self):
        # Set page title and layout
        st.set_page_config(page_title="🤖 " + self.config.get_page_title(), layout="wide")
        st.header("🤖 " + self.config.get_page_title())
        st.session_state.timeframe = ''
        st.session_state.IsFetchButtonClicked = False

        # Sidebar controls
        with st.sidebar:
            # Fetch configuration options
            llm_options = self.config.get_llm_options()
            usecase_options = self.config.get_usecase_options()

            # LLM selection dropdown
            self.user_controls["selected_llm"] = st.selectbox("Select LLM", llm_options)

            # Conditional input for Groq LLM API key and model selection
            if self.user_controls["selected_llm"] == "Groq":
                model_options = self.config.get_groq_model_options()
                self.user_controls["selected_groq_model"] = st.selectbox("Select Model", model_options)

                # Use session_state for API key persistence
                if "GROQ_API_KEY" not in st.session_state:
                    st.session_state["GROQ_API_KEY"] = ""

                self.user_controls["GROQ_API_KEY"] = st.text_input(
                    "API Key",
                    type="password",
                    value=st.session_state["GROQ_API_KEY"],
                    help="Get your API Key here: https://console.groq.com/keys"
                )
                st.session_state["GROQ_API_KEY"] = self.user_controls["GROQ_API_KEY"]

                # Display warning if API key is missing
                if not self.user_controls["GROQ_API_KEY"]:
                    st.warning("⚠️ Please enter your GROQ API key to proceed.")

            # Usecase selection dropdown
            self.user_controls["selected_usecase"] = st.selectbox("Select Usecase", usecase_options)
            
            if self.user_controls["selected_usecase"] == "Chatbot with Web Search Functionality" or self.user_controls["selected_usecase"] == "AI NEWS":
                os.environ["TAVILY_API_KEY"] = self.user_controls["TAVILY_API_KEY"] = st.session_state["TAVILY_API_KEY"] = st.text_input("TAVILY API KEY", type= "password")

                if not self.user_controls["GROQ_API_KEY"]:
                    st.warning("⚠️ Please enter your TAVILY_API_KEY key to proceed. Don't have ? refer : https://app.tavily.com")
                    
            if self.user_controls['selected_usecase'] == 'AI NEWS':
                st.subheader("🚀 AI NEWS EXPLORER")
                
                with st.sidebar:
                    time_frame = st.selectbox(
                        "🕒 Select Time Frame",
                        ["Daily", "Weekly", "Monthly"],
                        index=0
                    )

                if st.button("📰 Fetch Latest AI News", use_container_width=True):
                    st.session_state.IsFetchButtonClicked = True
                    st.session_state.timeframe = time_frame
                
        return self.user_controls