import os
from langchain_community.llms import DeepInfra
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
# Make sure to get your API key from DeepInfra. You have to Login and get a new token.
os.environ["DEEPINFRA_API_TOKEN"] = os.getenv("DEEPINFRA_API_TOKEN")

def gpt_4o(question):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a financial analyst. Answer user questions based on the given financial data."},
            {"role": "user", "content": question}
        ]
    )

    return completion.choices[0].message.content

def gpt_4(question):
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a financial analyst. Answer user questions based on the given financial data."},
            {"role": "user", "content": question}
        ]
    )

    return completion.choices[0].message.content

def gpt_o1(question):
    completion = client.chat.completions.create(
        model="gpt-o1",
        messages=[
            {"role": "system", "content": "You are a financial analyst. Answer user questions based on the given financial data."},
            {"role": "user", "content": question}
        ]
    )

    return completion.choices[0].message.content


def chunk_text(text, max_tokens=8000):
    words = text.split()
    for i in range(0, len(words), max_tokens):
        yield ' '.join(words[i:i + max_tokens])

def clean_text(text):
    return ' '.join(text.split())  # Removes extra spaces and newlines

def Meta_Llama_3_8B_Instruct(question):
    llm = DeepInfra(model_id="meta-llama/Meta-Llama-3-8B-Instruct")
    llm.model_kwargs = {
        "temperature": 0.7,
        "repetition_penalty": 1.2,
        "max_new_tokens": 250,
        "top_p": 0.9,
    }
    result = ""
    for chunk in chunk_text(question):
        result += llm.invoke(chunk) + "\n"
    return result

def Meta_Llama_3dot3_70B_Instruct_Turbo(question):
    llm = DeepInfra(model_id="meta-llama/Llama-3.3-70B-Instruct-Turbo")
    llm.model_kwargs = {
        "temperature": 0.7,
        "repetition_penalty": 1.2,
        "max_new_tokens": 100,  # Reduced to avoid context limit
        "top_p": 0.9,
    }
    question = clean_text(question)
    result = ""
    for chunk in chunk_text(question):
        result += llm.invoke(chunk) + "\n"
    return result

def Meta_Llama_3dot3_70B_Instruct(question):
    llm = DeepInfra(model_id="meta-llama/Llama-3.3-70B-Instruct")
    llm.model_kwargs = {
        "temperature": 0.7,
        "repetition_penalty": 1.2,
        "max_new_tokens": 100,
        "top_p": 0.9,
    }
    question = clean_text(question)
    result = ""
    for chunk in chunk_text(question):
        result += llm.invoke(chunk) + "\n"
    return result

def microsoft_phi_4(question):
    llm = DeepInfra(model_id="microsoft/phi-4")
    question = clean_text(question)
    result = ""
    for chunk in chunk_text(question):
        result += llm.invoke(chunk) + "\n"
    return result

def DeepSeek_R1(question):
    llm = DeepInfra(model_id="deepseek-ai/DeepSeek-R1")
    question = clean_text(question)
    result = ""
    for chunk in chunk_text(question):
        result += llm.invoke(chunk) + "\n"
    return result

def Mistral_Small_24B_Instruct_2501(question):
    llm = DeepInfra(model_id="mistralai/Mistral-Small-24B-Instruct-2501")
    question = clean_text(question)
    result = ""
    for chunk in chunk_text(question):
        result += llm.invoke(chunk) + "\n"
    return result


def ask_question(question, financials_text, model_name):
    """Send financial data and user question to the specified language model."""
    prompt = f"""
    The following is the financial data for a stock:

    {financials_text}

    Question: {question}
    Answer:
    """
    
    final_prompt = "You are a financial analyst. Answer user questions based on the given financial data." + prompt

    try:
        if model_name == "gpt-4o":
            result = gpt_4o(final_prompt)
        elif model_name == "gpt-4":
            result = gpt_4(final_prompt)
        elif model_name == "gpt-o1":
            result = gpt_o1(final_prompt)
        elif model_name == "Meta_Llama_3_8B_Instruct":
            result = Meta_Llama_3_8B_Instruct(final_prompt)
        elif model_name == "Meta_Llama_3dot3_70B_Instruct_Turbo":
            result = Meta_Llama_3dot3_70B_Instruct_Turbo(final_prompt)
        elif model_name == "Meta_Llama_3dot3_70B_Instruct":
            result = Meta_Llama_3dot3_70B_Instruct(final_prompt)
        elif model_name == "microsoft_phi_4":
            result = microsoft_phi_4(final_prompt)
        elif model_name == "DeepSeek_R1":
            result = DeepSeek_R1(final_prompt)
        elif model_name == "Mistral_Small_24B_Instruct_2501":
            result = Mistral_Small_24B_Instruct_2501(final_prompt)
        else:
            return f"Error: Model '{model_name}' not recognized."

        return result
    except Exception as e:
        return f"An error occurred: {str(e)}"
