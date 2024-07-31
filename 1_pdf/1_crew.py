from crewai import Agent, Crew, Process, Task
from crewai_tools import PDFSearchTool
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

llm=ChatOpenAI(model="gpt-4o-mini")

class Report(BaseModel):
    report: str = Field(..., description="Report")
    number_of_issues: int = Field(..., description="Number of issues found")



# --- Tools ---
# PDF SOURCE: https://www.gpinspect.com/wp-content/uploads/2021/03/sample-home-report-inspection.pdf
pdf_search_tool = PDFSearchTool(
    pdf="./example_home_inspection.pdf",
    config=dict(
        llm=dict(provider="openai", config=dict(model="gpt-4o-mini")),
    ),
)

# --- Agents ---
research_agent = Agent(
    role="Research Agent",
    goal="Search through the PDF to find relevant answers",
    allow_delegation=False,
    verbose=True,
    backstory=(
        """
        The research agent is adept at searching and 
        extracting data from documents, ensuring accurate and prompt responses.
        """
    ),
    tools=[pdf_search_tool],
    llm=llm
)

professional_writer_agent = Agent(
    role="Professional Writer",
    goal="Write professional emails based on the research agent's findings",
    allow_delegation=False,
    verbose=True,
    backstory=(
        """
        The professional writer agent has excellent writing skills and is able to craft 
        clear and concise emails based on the provided information.
        """
    ),
    tools=[],
    llm=llm
)

# --- Tasks ---
answer_customer_question_task = Task(
    description=(
        """
        Answer the customer's questions based on the home inspection PDF.
        The research agent will search through the PDF to find the relevant answers.
        Your final answer MUST be clear and accurate, based on the content of the home
        inspection PDF.

        Here is the customer's question:
        {customer_question}
        """
    ),
    expected_output="""
        Provide clear and accurate answers to the customer's questions based on 
        the content of the home inspection PDF.
        """,
    tools=[pdf_search_tool],
    agent=research_agent,
)

write_email_task = Task(
    description=(
        """
        - Write a professional email to a contractor based 
            on the research agent's findings.
        - The email should clearly state the issues found in the specified section 
            of the report and request a quote or action plan for fixing these issues.
        - Ensure the email is signed with the following details:
        
            Best regards,

            Brandon Hancock,
            Hancock Realty
        """
    ),
    expected_output="""
        Write a clear and concise email that can be sent to a contractor to address the 
        issues found in the home inspection report.
        """,
    tools=[],
    agent=professional_writer_agent,
    output_pydantic=Report
)

# --- Crew ---
crew = Crew(
    agents=[research_agent, professional_writer_agent],
    tasks=[answer_customer_question_task, write_email_task],
    process=Process.sequential,
)

# customer_question = input(
#     "Which section of the report would you like to generate a work order for?\n"
# )

result = crew.kickoff(inputs={"customer_question": "Roof"})

# result = crew.kickoff_for_each(inputs=[{"customer_question": "Roof"}, {"customer_question": "Eletrical"}])


# print("===============================")
# print(result.raw)
# print("===============================")
# print(result.pydantic)
# print("===============================")
# print(result.tasks_output)
# print("===============================")
# print(result.token_usage)
# print("===============================")
print(result)
