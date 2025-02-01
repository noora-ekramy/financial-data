import streamlit as st
import pandas as pd
from yahooquery import Ticker
from openai import OpenAI

# Set up OpenAI client
client = OpenAI(api_key="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")
# Function to fetch financial data
def get_financial_data(ticker):
    """Fetch financial statements for a given stock ticker."""
    stock = Ticker(ticker)
    income_statement = stock.income_statement()
    balance_sheet = stock.balance_sheet()
    cash_flow = stock.cash_flow()
    return income_statement, balance_sheet, cash_flow

# Function to send data + question to OpenAI API
def ask_question(question, financials_text):
    """Send financial data and user question to OpenAI API."""
    prompt = f"""
    The following is the financial data for a stock:

    {financials_text}

    Question: {question}
    Answer:
    """
    
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a financial analyst. Answer user questions based on the given financial data."},
            {"role": "user", "content": prompt}
        ]
    )

    return completion.choices[0].message.content

# Streamlit UI
st.title("📊 Stock Financial Analysis")

# Input for stock ticker
ticker = st.text_input("Enter Stock Ticker (e.g., NVDA, AAPL, TSLA):")

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

    # Button to send data & question to OpenAI API
    if st.button("Analyze Data"):
        if user_question:
            answer = ask_question(user_question, st.session_state.financials_text)
            st.subheader("📢 Analysis")
            st.write(answer)
        else:
            st.warning("⚠️ Please enter a question before clicking 'Analyze Data'.")
else:
    st.info("📌 Please load financial data before asking a question.")
