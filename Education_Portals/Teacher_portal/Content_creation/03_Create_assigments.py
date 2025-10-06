import os
import json
import asyncio
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List , Optional
from langchain_community.document_loaders import PyPDFLoader
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel

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

llm_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)

# ----------------------- Streamlit UI -----------------------
st.set_page_config(page_title="PDF/Text Viewer", layout="wide")
st.title("📘 Upload PDF or Paste Text")

uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
pasted_text = st.text_area("Or paste text here", height=200)

# -------------------- Process PDF --------------------
def process_pdf(uploaded_files, pasted_text):
    all_text = ""
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
    if pasted_text:
        all_text += "\n\n" + pasted_text.strip()
    return all_text.strip()

# -------------------- Ensure pdf_text is in session state --------------------
if "pdf_text" not in st.session_state:
    st.session_state["pdf_text"] = ""

if uploaded_files or pasted_text:
    st.session_state["pdf_text"] = process_pdf(uploaded_files, pasted_text)

# -------------------- Models --------------------
class Question(BaseModel):
    question_text: str
    options: List[str]

class Quiz(BaseModel):
    title: str
    questions: List[Question]

class ExamQuestion(BaseModel):
    question_text: str
    answer: Optional[str] = None  # placeholder for user input

class Exam(BaseModel):
    title: str
    questions: List[ExamQuestion]
# -------------------- Agents --------------------
quiz_agent: Agent = Agent(
    name="quiz_generator_agent",
    instructions="Provide multiple choice questions with options from the text.",
    output_type=Quiz,
    model=llm_model
)
exam_agent: Agent = Agent(
    name="exam_generator_agent",
    instructions="Provide open-ended exam questions based on the text.",
    output_type=Exam,
    model=llm_model
)

async def agent_output(agent_instance, input_text):
    result_output = await Runner.run(agent_instance, input_text)
    return result_output.final_output

# -------------------- Session & Buttons --------------------
if "show_quiz" not in st.session_state:
    st.session_state["show_quiz"] = False
if "show_exam" not in st.session_state:
    st.session_state["show_exam"] = False

col1, col2 = st.columns(2)
with col1:
    if st.button("Quiz"):
        if not st.session_state["pdf_text"]:
            st.warning("⚠️ Upload PDF or paste text first!")
        else:
            st.session_state["show_quiz"] = True
            st.session_state["show_exam"] = False

with col2:
    if st.button("Exam"):
        if not st.session_state["pdf_text"]:
            st.warning("⚠️ Upload PDF or paste text first!")
        else:
            st.session_state["show_exam"] = True
            st.session_state["show_quiz"] = False

# -------------------- Quiz Workflow --------------------
if st.session_state["show_quiz"] and st.session_state["pdf_text"]:
    if "quiz_assignment" not in st.session_state:
        st.session_state["quiz_assignment"] = asyncio.run(agent_output(quiz_agent, st.session_state["pdf_text"]))

    quiz_assignment: Quiz = st.session_state["quiz_assignment"]
    st.title(quiz_assignment.title)

    if "user_answers" not in st.session_state:
        st.session_state["user_answers"] = {}

    with st.form("quiz_form"):
        for idx, question in enumerate(quiz_assignment.questions, start=1):
            st.markdown(f"**Q{idx}: {question.question_text}**")
            selected_option = st.radio(
                label=f"Select your answer for Q{idx}",
                options=question.options,
                key=f"q{idx}"
            )
            st.session_state["user_answers"][f"Q{idx}"] = {
                "Q": question.question_text,
                "options": question.options,
                "user_select": selected_option
            }
        submitted = st.form_submit_button("Submit Quiz")
        if submitted:
            st.success("✅ Quiz Submitted! Here are your selections:")
            st.json(st.session_state["user_answers"])

# -------------------- Exam Workflow --------------------
if st.session_state["show_exam"] and st.session_state["pdf_text"]:
    if "exam_assignment" not in st.session_state:
        st.session_state["exam_assignment"] = asyncio.run(agent_output(exam_agent, st.session_state["pdf_text"]))

    exam_assignment: Exam = st.session_state["exam_assignment"]
    st.title(exam_assignment.title)

    if "exam_answers" not in st.session_state:
        st.session_state["exam_answers"] = {}

    with st.form("exam_form"):
        for idx, question in enumerate(exam_assignment.questions, start=1):
            st.markdown(f"**Q{idx}: {question.question_text}**")
            # Use empty string if answer is None
            user_input = st.text_area(
                label=f"Your Answer for Q{idx}",
                value=question.answer or "",
                key=f"exam_q{idx}"
            )
            st.session_state["exam_answers"][f"Q{idx}"] = {
                "Q": question.question_text,
                "user_answer": user_input
            }

        submitted = st.form_submit_button("Submit Exam")
        if submitted:
            st.success("✅ Exam Submitted! Here are your answers:")
            st.json(st.session_state["exam_answers"])
