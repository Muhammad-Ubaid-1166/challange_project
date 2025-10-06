from ast import List
import os
import asyncio
import json
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel , Field 
from dataclasses import dataclass
from langchain_community.document_loaders import PyPDFLoader
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled  , AgentOutputSchema


# -----------------------
# Load env and API key
# -----------------------
load_dotenv()
with open("config.json") as f:
    config = json.load(f)

gemini_api_key = config.get("GEMINI_KEY") or os.environ.get("GEMINI_KEY")
if not gemini_api_key:
    st.error("❌ Missing GEMINI_KEY in config.json or environment")
    st.stop()
external_client: AsyncOpenAI = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# 2. Which LLM Model?
llm_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="PDF/Text Viewer", layout="wide")
st.title("📘 Upload PDF or Paste Text")

st.write("Upload one or more PDFs, or paste text below:")

# ✅ Allow multiple PDF uploads
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
pasted_text = st.text_area("Or paste text here", height=200)

process_btn = st.button("Show Content")


# ✅ Updated to handle multiple PDFs
def process_pdf(uploaded_files, pasted_text):
    all_text = ""

    # Process uploaded PDF files
    if uploaded_files:
        for i, uploaded_file in enumerate(uploaded_files, start=1):
            pdf_path = f"temp_{i}.pdf"
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            if not docs:
                st.error(f"⚠️ No pages loaded from {uploaded_file.name}.")
                continue

            full_text = "\n\n".join(
                f"--- {uploaded_file.name.upper()} | PAGE {j+1} ---\n{doc.page_content.strip()}"
                for j, doc in enumerate(docs)
            )

            all_text += "\n\n" + full_text

    # Add pasted text if provided
    if pasted_text:
        all_text += "\n\n" + pasted_text.strip()

    return all_text.strip()


from pydantic import BaseModel, Field
from typing import List, Dict

# -------------------- Data Models -------------------- #
class Unit(BaseModel):
    unit_name: str = Field(..., description="Name of the unit or chapter.")
    page: int = Field(..., description="Starting page number of the unit.")

class Subject(BaseModel):
    subject_name: str = Field(..., description="Name of the subject, e.g., English, Math, Computer.")
    units: List[Unit] = Field(..., description="List of units within the subject.")

class FinalOutputType(BaseModel):
    subjects: List[Subject] = Field(..., description="List of subjects with their respective units.")

class TeacherAndStudentObligation(BaseModel):
    final_index_and_chapters: FinalOutputType = Field(..., description="The full subject/unit structure.")
    unit: List[Dict[str, str]] = Field(..., description="Chapter name -> important quote for the teacher.")
    all_book_teaching_method: str = Field(..., description="Overall teaching method for the book.")

# -------------------- Agent Setup -------------------- #
agent: Agent = Agent(
    name="helpful_assistant",
    instructions="Just give structured output from the data provided by the user.",
    output_type=AgentOutputSchema(TeacherAndStudentObligation,strict_json_schema=False),
    model=llm_model
)

 # gemini-2.5 as agent brain - chat completions

  
async def agent_output():
    result_output = await Runner.run(
        agent,
        result
    )
    return result_output.final_output


if process_btn:
    result = process_pdf(uploaded_files, pasted_text)
    # Fetch structured output from the agent
    teacher_obligation = asyncio.run(agent_output())

    # Display overall teaching method
    st.markdown(f"📝 **Teaching Method:** {teacher_obligation.all_book_teaching_method}\n")

    # -------------------- Subjects & Units --------------------
    for subject in teacher_obligation.final_index_and_chapters.subjects:
        st.markdown(f"### 📚 {subject.subject_name}")  # Subject heading

        # Display units
        for i, unit in enumerate(subject.units, start=1):
            st.markdown(f"🔹 Unit {i}: {unit.unit_name} (Page: {unit.page})")

    st.markdown("---")

    # -------------------- Teacher Notes --------------------
    st.subheader("💡 Teacher Notes")
    # Loop over subjects to group notes
    for subject in teacher_obligation.final_index_and_chapters.subjects:
        st.markdown(f"### 📚 {subject.subject_name}")  # Subject heading for notes

        for unit in subject.units:
            # Find the teacher quote for this unit
            quote = ""
            for chapter_dict in teacher_obligation.unit:
                if unit.unit_name in chapter_dict:
                    quote = chapter_dict[unit.unit_name]
                    break

            # Display teacher note with unit name
            st.markdown(f"**{unit.unit_name}:** {quote}")
            st.markdown("In this unit, you have to teach students etc.\n")
