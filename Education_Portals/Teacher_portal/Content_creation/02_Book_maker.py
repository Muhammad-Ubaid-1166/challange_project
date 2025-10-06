import os
import json
import streamlit as st
from typing import List, Dict
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib import colors

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

# -----------------------
# Schema
# -----------------------
class Page(BaseModel):
    page: int
    chapter: str = Field(
        default="",
        description="The title of the main chapter or section the page belongs to.",
    )
    headings: List[str] = Field(
        default_factory=list,
        description="A list of 1 to 3 major subsection headings found on the page.",
        max_length=3
    )
    Chapter_theory: str = Field(
        description="The complete long chapter theory of the page.",
    
    )

class DocumentOutput(BaseModel):
    index: Dict[int, str]
    pages: List[Page]
    About_book: str = Field(
        description="A high-level summary or introduction for the entire document."
        
    )

# -----------------------
# LLM init
# -----------------------
model = ChatOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    model="gemini-2.5-flash",
    temperature=0,
)

structured_model = model.with_structured_output(DocumentOutput)

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="AI Document Parser", layout="wide")
st.title("📘 AI Document Parser")

st.write("Upload a PDF or paste raw text below:")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
pasted_text = st.text_area("Or paste text here", height=200)

process_btn = st.button("Process Document")

if process_btn:
    if uploaded_file is not None:
        # Save uploaded PDF temporarily
        pdf_path = "temp.pdf"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        if not docs:
            st.error("No pages loaded from PDF.")
            st.stop()

        full_text = "\n\n".join(
            f"---PAGE {i+1}---\n{doc.page_content.strip()}"
            for i, doc in enumerate(docs)
        )

    elif pasted_text.strip():
        pages = pasted_text.strip().split("\n\n")
        full_text = "\n\n".join(
            f"---PAGE {i+1}---\n{p}" for i, p in enumerate(pages)
        )

    else:
        st.warning("⚠️ Please upload a PDF or paste text.")
        st.stop()

    # -----------------------
    # Prompt
    # -----------------------
    prompt = PromptTemplate(
        template=(
            "You are given the full text of a document below. "
            "Produce a JSON object that matches this schema:\n\n"
            "DocumentOutput(\n"
            "  index: dict[int, str],\n"
            "  pages: list of objects with keys: page (int), chapter (string), headings (list of strings), Chapter_theory (string)\n"
            ")\n\n"
            "Requirements:\n"
            "- Return data only in the required structure (no extra fields).\n"
            "- `index` should map page numbers to chapter titles.\n"
            "- For each page, provide: page, chapter, headings, and Chapter_theory.\n"
            "- The `Chapter_theory` must be a long, continuous text filling the page, around 200 words maximum per page.\n"
            "- Ensure each page has well-structured and meaningful content, not bullet points.\n"
            "- Expand content naturally if the source text is too short.\n\n"
            "Document:\n\n{text}\n\n"
            "Now generate the structured syllabus with detailed page content."
        ),
        input_variables=["text"]
    )
    long_text_generator = PromptTemplate(
        template=(
            "you are a chapter_theory_writer here is the chapter_theory text : {chapter_theory} "
            "write a long, continuous text filling the page, around 500 words maximum per page."
            
        ),
        input_variables=["chapter_theory"]
    )

    prompt_text = prompt.format(text=full_text)

    with st.spinner("⏳ Processing with LLM..."):
        structured_output = structured_model.invoke(prompt_text)

    # -----------------------
    # Display Output
    # -----------------------
    st.subheader("📑 Document Index")
    chapter_start_page = {}
    for page in structured_output.pages:
        if page.chapter not in chapter_start_page:
            chapter_start_page[page.chapter] = page.page

    for idx, (chapter, start_page) in enumerate(chapter_start_page.items(), start=1):
        st.write(f"**{idx}. {chapter}** (starts at page {start_page})")

    st.subheader("📖 About the Book")
    st.write(structured_output.About_book)

    st.subheader("📄 Pages")
    for page in structured_output.pages:
        with st.expander(f"Page {page.page}: {page.chapter}"):
            if page.headings:
                st.write(f"**Heading:** {page.headings[0]}")
            if len(page.Chapter_theory) < 500:
                long_text_generator = long_text_generator.format(chapter_theory=page.Chapter_theory)
                long_text_output = long_text_generator.invoke(long_text_generator)
                st.write(long_text_output)
            else:
                st.write(page.Chapter_theory)

# _________________c___________________





    # -----------------------
    # PDF Generation Function
    # -----------------------
    def generate_structured_pdf(structured_output):
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=A4,
            rightMargin=50, leftMargin=50, topMargin=60, bottomMargin=50
        )
        styles = getSampleStyleSheet()

        # Custom Styles
        title_style = ParagraphStyle(
            "TitleStyle",
            parent=styles["Heading1"],
            fontSize=20,
            leading=26,
            alignment=1,
            spaceAfter=30,
            textColor=colors.HexColor("#2E4053")
        )
        chapter_style = ParagraphStyle(
            "ChapterStyle",
            parent=styles["Heading2"],
            fontSize=16,
            leading=22,
            spaceBefore=20,
            spaceAfter=12,
            textColor=colors.HexColor("#1F618D")
        )
        heading_style = ParagraphStyle(
            "HeadingStyle",
            parent=styles["Heading3"],
            fontSize=13,
            leading=18,
            spaceBefore=10,
            spaceAfter=6,
            textColor=colors.HexColor("#117864")
        )
        body_style = ParagraphStyle(
            "BodyStyle",
            parent=styles["Normal"],
            fontSize=11,
            leading=16,
            spaceBefore=4,
            spaceAfter=8,
            alignment=4,  # justify
        )

        story = []

        # Title Page
        story.append(Spacer(1, 100))
        story.append(Paragraph("📘 Structured Document Output", title_style))
        story.append(Spacer(1, 40))

        # About the Book
        story.append(Paragraph("About the Book", chapter_style))
        story.append(Paragraph(structured_output.About_book, body_style))
        story.append(PageBreak())

        # Index Section
        story.append(Paragraph("Table of Contents", chapter_style))
        for idx, (chapter, start_page) in enumerate(structured_output.index.items(), start=1):
            story.append(Paragraph(f"<b>{idx}. {chapter}</b> — starts at Page {start_page}", body_style))
        story.append(PageBreak())

        # Content Pages
        for page in structured_output.pages:
            story.append(Paragraph(f"Page {page.page}: {page.chapter}", chapter_style))
            if page.headings:
                story.append(Paragraph(f"{page.headings[0]}", heading_style))
            story.append(Paragraph(page.Chapter_theory, body_style))
            story.append(PageBreak())

        # Footer with page numbers
        def add_page_number(canvas, doc):
            page_num = canvas.getPageNumber()
            text = f"Page {page_num}"
            canvas.setFont("Helvetica", 9)
            canvas.setFillColor(colors.grey)
            canvas.drawRightString(A4[0] - 40, 20, text)

        doc.build(story, onLaterPages=add_page_number, onFirstPage=add_page_number)

        buffer.seek(0)
        return buffer

    # -----------------------
    # Show Download Button
    # -----------------------
    pdf_buffer = generate_structured_pdf(structured_output)

    st.download_button(
        label="📥 Download Structured PDF",
        data=pdf_buffer,
        file_name="structured_document.pdf",
        mime="application/pdf"
    )
