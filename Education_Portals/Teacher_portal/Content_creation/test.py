import os
import json
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel , Field 
from dataclasses import dataclass
from langchain_community.document_loaders import PyPDFLoader
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled , RunContextWrapper 



with open("config.json") as f:
    config = json.load(f)

gemini_api_key = config.get("GEMINI_KEY") or os.environ.get("GEMINI_KEY")


set_tracing_disabled(disabled=True)

# 1. Which LLM Service?
external_client: AsyncOpenAI = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# 2. Which LLM Model?
llm_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)
def test():
    result = """ 
    The Rise of Autonomous AI: Understanding Agentic Systems
    The field of Artificial Intelligence (AI) is rapidly evolving beyond passive data analysis and reactive content generation. The latest frontier, known as Agentic AI, marks a significant shift, endowing systems with autonomy, the capacity to plan, and the ability to execute complex, multi-step goals with minimal human oversight. Agentic AI systems are defined by their capacity for self-directed action, utilizing core components like Large Language Models (LLMs) as their "brain" to orchestrate tasks and make contextual decisions. This progression from traditional, rules-based software to truly autonomous agents promises to unlock unprecedented efficiency across industries, yet it simultaneously introduces profound ethical and operational challenges that humanity must address responsibly.

    At its core, agentic behavior is characterized by a continuous, adaptive loop of Perception, Planning, Action, and Reflection. Unlike a simple chatbot that reacts to a single prompt, an agentic system can perceive a high-level objective, such as "research and book a flight to Paris for under $500." It then breaks this task down into sequential sub-tasks: searching for flights, checking price trends, comparing airports, and ultimately interfacing with booking APIs to execute the purchase. The crucial element is Reflection, where the agent evaluates the success of its previous actions and adjusts its future steps—a process often referred to as a "data flywheel." If a booking fails, the agent doesn't stop; it reasons about the failure and attempts an alternative strategy, thereby learning and improving its performance iteratively over time.

    The transformative potential of Agentic AI is already manifesting in complex domains. In software engineering, autonomous agents are automating large portions of the development lifecycle, including code generation, testing, and debugging, thereby accelerating innovation. In supply chain management, agentic systems analyze real-time data from inventory, logistics, and demand forecasting to dynamically optimize delivery routes and production schedules, reducing costly bottlenecks. Furthermore, in financial services, agents are being deployed for sophisticated fraud detection and automated, context-aware risk assessments. These applications illustrate the shift from AI as a mere tool to AI as a collaborative partner capable of managing end-to-end business processes.

    However, the introduction of autonomous agents with high-level decision-making capacity amplifies existing concerns about AI governance. The "black box" nature of complex LLMs makes the agents’ reasoning pathways less transparent, complicating the establishment of accountability when an error or unintended consequence occurs. Moreover, an autonomous agent's ability to operate in dynamic, real-world environments increases the risk of misalignment between the agent's predefined goals and human values. Addressing issues of fairness, mitigating bias embedded in training data, and ensuring reliable human oversight—or the "human-in-the-loop" mechanism—are paramount to preventing systemic risks and maintaining societal trust.

    In conclusion, Agentic AI represents a technological leap forward, moving systems from mere computation to genuine autonomy and initiative. The key to successfully integrating this technology lies not only in maximizing its inherent benefits—such as increased efficiency and innovation—but also in rigorously developing robust ethical frameworks and strong technical guardrails. As AI gains agency, the human responsibility shifts from constant guidance to careful, proactive governance, ensuring that these powerful autonomous systems remain aligned with the best interests of society.
    """
    return result
function_result = test()
@dataclass
class data:
    text:str
def dynamic_instructions(wrapper: RunContextWrapper[data], agent: Agent[data]) -> str:
    "you just have to show the wrapper inner value"
    wrapper = wrapper.context.text
   
    return wrapper
class final_output_type(BaseModel):
    text:str
agent: Agent = Agent(name="helpful_assistent",
                     instructions=dynamic_instructions,
                     output_type=final_output_type,
                     model=llm_model) # gemini-2.5 as agent brain - chat completions
test: Agent = Agent(name="helpful_assistent",
                     instructions=dynamic_instructions,
                     output_type=final_output_type,
                     model=llm_model) # gemini-2.5 as agent brain - chat completions
output = data(function_result)
result_output: Runner = Runner.run_sync(agent, "use your context and tell me which notes you have",context=output)

print("\nCALLING AGENT\n")
print(result_output.final_output)


# from typing import Dict
# from pydantic import BaseModel # Changed from RootModel

# # Define the Pydantic model using BaseModel.
# # Since the original dictionary keys are dynamic, the entire mapping 
# # must be placed inside a single, explicit field (here, named 'units').
# class UnitMapping(BaseModel):
#     """
#     A simple Pydantic model containing a dictionary mapping unit names (string) 
#     to scores (integer) as a field called 'units'.
#     """
#     units: Dict[str, int] # The dictionary is now a named field

# # --- Simple Usage Example ---

# # 1. The data now conforming to the BaseModel structure
# # The unit dictionary is wrapped under the 'units' key to match the BaseModel field.
# data = {
#     "units": {
#         "unit 1 : Agentic ai": 1,
#         "unit 2 : Generative ai": 35
#     }
# }

# # 2. Validate the data
# # Pydantic ensures the input contains the 'units' field with the correct dictionary type.
# validated_mapping = UnitMapping.model_validate(data)

# # 3. Print the simple output, accessing the validated dictionary using the .units attribute
# print(validated_mapping.units)
# for unit_name, score in validated_mapping.units.items():
#     # unit_name is the string, score is the integer
#     print(f"{unit_name} : {score}")
# Output is a Python dictionary:
# {'unit 1 : Agentic ai': 1, 'unit 2 : Generative ai': 35}


import streamlit as st
from pydantic import BaseModel, Field
from typing import List

# -------------------- Pydantic Models -------------------- #
class Question(BaseModel):
    question_text: str
    options: List[str]

class Quiz(BaseModel):
    title: str
    questions: List[Question]

# -------------------- Sample Quiz -------------------- #
sample_quiz = Quiz(
    title="Agentic AI Quiz",
    questions=[
        Question(
            question_text="What is agentic AI?",
            options=["Option 1", "Option 2", "Option 3"]
        ),
        Question(
            question_text="Which one is an example of agentic AI?",
            options=["ChatGPT", "Spreadsheet", "Calculator"]
        )
    ]
)

# -------------------- Streamlit UI -------------------- #
st.title(sample_quiz.title)

# Dictionary to store user's selections
user_answers = {}

# Iterate through questions
for idx, question in enumerate(sample_quiz.questions, start=1):
    st.markdown(f"**Q{idx}: {question.question_text}**")

    # Use radio buttons so user can select one option
    selected_option = st.radio(
        label=f"Select your answer for Q{idx}",
        options=question.options,
        key=f"q{idx}"
    )

    # Store the question, options, and user selection
    user_answers[f"Q{idx}"] = {
        "Q": question.question_text,
        "options": question.options,
        "user_select": selected_option
    }

st.markdown("---")

# Button to submit answers
if st.button("Submit Quiz"):
    st.success("✅ Quiz Submitted! Here are your selections:")
    st.json(user_answers)
