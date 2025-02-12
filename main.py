import streamlit as st
import pandas as pd
from yahooquery import Ticker
from openai import OpenAI
import os
from dotenv import load_dotenv
from financial_model_api import *
# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
# Set up OpenAI client
client = OpenAI(api_key=api_key)

# Function to fetch financial data
def get_financial_data(ticker):
    """Fetch financial statements for a given stock ticker."""
    stock = Ticker(ticker)
    income_statement = stock.income_statement()
    balance_sheet = stock.balance_sheet()
    cash_flow = stock.cash_flow()
    return income_statement, balance_sheet, cash_flow

# Streamlit UI

st.markdown("""
    <style>
    html, body, .stApp {
        background-color: #041317; 
        height: 100%;
        margin: 0;
        padding: 0;
    }
    .stApp {
        display: flex;
        flex-direction: column;
        min-height: 100vh;
    }
    .tagline {
        font-size: 20px;
        font-weight: bold;
        color: #a4ffff;
        text-align: center;
        margin-top: -10px;
    }
    .subtagline {
        font-size: 14px;
        color: #fcfcfc;
        text-align: center;
        padding:20px;
        margin-bottom: 20px;
    }
     .header {
        font-size: 39px;
        font-weight: bold;
        color: #65daff;
        text-align: center;
        margin-top: -10px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

col1, col2 = st.columns([1, 5]) 

with col1:
    st.image("youtiva-logo.png", width=100)

with col2:
    st.title("Youtiva")

st.markdown('<div class="tagline">Stand Out & Excel with Your Unique AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtagline">Empowering businesses with tailored AI solutions to streamline operations, boost efficiency, and sustain competitive advantage</div>', unsafe_allow_html=True)
st.markdown('<div class="header">Financial Statements</div>', unsafe_allow_html=True)




# Input for stock ticker
ticker = st.text_input("Enter Financial Ticker (e.g., NVDA, AAPL, TSLA):")

# Model selection dropdown
model_name = st.selectbox(
    "Choose a model:",
    [
        "gpt-4o", 
        "Meta_Llama_3_8B_Instruct", 
        "Meta_Llama_3dot3_70B_Instruct_Turbo", 
        "Meta_Llama_3dot3_70B_Instruct", 
        "Mistral_Small_24B_Instruct_2501"
    ],
    index=0  # Default to gpt-4o
)

# Initialize session state to keep financial data sticky
if "financials_loaded" not in st.session_state:
    st.session_state.financials_loaded = False
    st.session_state.income_df = None
    st.session_state.balance_df = None
    st.session_state.cash_df = None
    st.session_state.financials_text = ""

# Button to fetch financial data
if st.button("Show Financial Data"):
    if ticker:
        # Fetch financial data
        income, balance, cash = get_financial_data(ticker.upper())

        # Convert to DataFrame (handling API response)
        if isinstance(income, list): income = pd.DataFrame(income)
        if isinstance(balance, list): balance = pd.DataFrame(balance)
        if isinstance(cash, list): cash = pd.DataFrame(cash)

        # Store data in session state to keep tables sticky
        st.session_state.financials_loaded = True
        st.session_state.income_df = income
        st.session_state.balance_df = balance
        st.session_state.cash_df = cash

        # Convert financial data to text for AI analysis
        st.session_state.financials_text = (
            f"Income Statement:\n{income.to_string()}\n\n"
            f"Balance Sheet:\n{balance.to_string()}\n\n"
            f"Cash Flow Statement:\n{cash.to_string()}"
        )

# Always display tables if data is loaded (makes them sticky)
if st.session_state.financials_loaded:
    st.subheader("📜 Income Statement")
    st.dataframe(st.session_state.income_df, height=300)

    st.subheader("📊 Balance Sheet")
    st.dataframe(st.session_state.balance_df, height=300)

    st.subheader("💰 Cash Flow Statement")
    st.dataframe(st.session_state.cash_df, height=300)

    # Input box for user question
    user_question = st.text_area("Ask a question about this financial data:")

    # Button to send data & question to the selected model
    if st.button("Analyze Data"):
        if user_question:
              # Import the updated function
            answer = ask_question(user_question, st.session_state.financials_text, model_name)
            st.subheader("📢 Analysis")
            st.write(answer)
        else:
            st.warning("⚠️ Please enter a question before clicking 'Analyze Data'.")
else:
    st.info("📌 Please load financial data before asking a question.")
