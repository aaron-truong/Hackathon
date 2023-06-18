import os
import random
from collections import deque
from typing import Dict, List, Optional, Any, Callable

from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field
from langchain.chains.base import Chain

from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
import faiss

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)

from dialogue_template import DialogueAgent, DialogueSimulator


api_key = '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'

company = "Jane Street"
comp_descrip = '''
Jane Street is a global proprietary trading firm that specializes in quantitative trading and operates 
in various financial markets. It was founded in 2000 and is headquartered in New York City. Jane Street 
is known for its expertise in algorithmic trading and high-frequency trading strategies.
'''

self_descrip = '''
Objective:
Highly motivated and skilled Computer Science Masters student seeking a challenging position to leverage my technical expertise and academic accomplishments at UC Berkeley. Passionate about applying cutting-edge technologies and solving complex problems in the field of computer science.

Education:
Master of Science in Computer Science, University of California, Berkeley

Bachelor of Science in Computer Science, University of California, Berkeley

Relevant Coursework:

Advanced Algorithms
Artificial Intelligence
Machine Learning
Data Structures
Database Systems
Computer Networks
Operating Systems
Software Engineering

Skills:

Programming Languages: Python, Java, C++, JavaScript
Frameworks and Libraries: TensorFlow, PyTorch, Django, React
Database Management: SQL, MongoDB, PostgreSQL
Data Analysis and Visualization: Pandas, NumPy, Matplotlib, Tableau
Web Development: HTML, CSS, JavaScript, Node.js
Version Control: Git, GitHub
Operating Systems: Windows, Linux, macOS
Problem Solving and Algorithm Design
Strong Mathematical Foundation
'''

job_title = 'Software Engineering Intern'
job_descrip = '''We are looking for Software Engineering Interns to join our Winter 2024 Co-op program. Our goal is to give you a real sense of what it’s like to work at Jane Street full time. Over the course of your internship, you will explore ways to approach and solve exciting problems within your field of interest through fun and challenging classes, interactive sessions, and group discussions—and then you will have the chance to put those lessons to practical use.

As an intern, you are paired with full-time employees who act as mentors, collaborating with you on real-world projects we actually need done. When you’re not working on your project, you will have plenty of time to use our office amenities, physical and virtual educational resources, attend social events, and engage with the parts of our work that excite you the most.

If you’ve never thought about a career in finance, you’re in good company. Many of us were in the same position before working here. If you have a curious mind, a collaborative spirit, and a passion for solving interesting problems, we have a feeling you’ll fit right in.

As a Software Engineering intern, you’ll learn how we use OCaml (our primary development language) in our day to day work, and gain exposure to the libraries and tools that are foundational to our internal systems.

During the program you’ll work on two projects, mentored closely by the full-time employees who designed them. Some projects consider big-picture questions that we’re still trying to figure out, while others involve building something new. Your mentors will work in two distinct areas, so you’ll gain a better understanding of the wide range of problems we solve every day, from high performance trading systems to programming language design and everything in between. 

If you’d like to learn more, you can read about our interview and team placement processes and get a sense of what our most recent intern projects looked like.

About You
We don’t expect you to have a background in finance, OCaml, functional programming, or any other specific field—we’re looking for smart people who enjoy solving interesting problems. We’re more interested in how you think and learn than what you currently know. You should be:

Must be enrolled in a Co-op program at your university
A top-notch programmer with a passion for technology
Collaborative and courteous with strong interpersonal and communication skills
Eager to ask questions, admit mistakes, and learn new things
Must be able to work in-person in our NYC office from January 2024 through April 2024
Fluent in English'''

rng = 1.0
focus = "technical"
role = "Senior Trader"
name = "Nick"
my_role = "college student"

# Define your embedding model
embeddings_model = OpenAIEmbeddings(openai_api_key=api_key)

llm = OpenAI(openai_api_key=api_key, temperature=rng)
chat_llm = ChatOpenAI(openai_api_key=api_key, temperature=rng)

embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

class TaskCreationChain(LLMChain):
    """Chain to generates tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_creation_template = (
            "You are a task creation AI that uses the result of an execution agent"
            " to create new tasks with the following objective: {objective},"
            " The last completed task has the result: {result}."
            " This result was based on this task description: {task_description}."
            " These are incomplete tasks: {incomplete_tasks}."
            " Based on the result, create new tasks to be completed"
            " by the AI system that do not overlap with incomplete tasks."
            " Return the tasks as an array."
        )
        prompt = PromptTemplate(
            template=task_creation_template,
            input_variables=[
                "result",
                "task_description",
                "incomplete_tasks",
                "objective",
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)

class TaskPrioritizationChain(LLMChain):
    """Chain to prioritize tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_prioritization_template = (
            "You are a task prioritization AI tasked with cleaning the formatting of and reprioritizing"
            " the following tasks: {task_names}."
            " Consider the ultimate objective of your team: {objective}."
            " Do not remove any tasks. Return the result as a numbered list, like:"
            " #. First task"
            " #. Second task"
            " Start the task list with number {next_task_id}."
        )
        prompt = PromptTemplate(
            template=task_prioritization_template,
            input_variables=["task_names", "next_task_id", "objective"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)

class ExecutionChain(LLMChain):
    """Chain to execute tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        execution_template = (
            "You are an AI who performs one task based on the following objective: {objective}."
            " Take into account these previously completed tasks: {context}."
            " Your task: {task}."
            " Response:"
        )
        prompt = PromptTemplate(
            template=execution_template,
            input_variables=["objective", "context", "task"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)

def get_next_task(
    task_creation_chain: LLMChain,
    result: Dict,
    task_description: str,
    task_list: List[str],
    objective: str,
) -> List[Dict]:
    """Get the next task."""
    incomplete_tasks = ", ".join(task_list)
    response = task_creation_chain.run(
        result=result,
        task_description=task_description,
        incomplete_tasks=incomplete_tasks,
        objective=objective,
    )
    new_tasks = response.split("\n")
    return [{"task_name": task_name} for task_name in new_tasks if task_name.strip()]

def prioritize_tasks(
    task_prioritization_chain: LLMChain,
    this_task_id: int,
    task_list: List[Dict],
    objective: str,
) -> List[Dict]:
    """Prioritize tasks."""
    task_names = [t["task_name"] for t in task_list]
    next_task_id = int(this_task_id) + 1
    response = task_prioritization_chain.run(
        task_names=task_names, next_task_id=next_task_id, objective=objective
    )
    new_tasks = response.split("\n")
    prioritized_task_list = []
    for task_string in new_tasks:
        if not task_string.strip():
            continue
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = task_parts[0].strip()
            task_name = task_parts[1].strip()
            prioritized_task_list.append({"task_id": task_id, "task_name": task_name})
    return prioritized_task_list

def _get_top_tasks(vectorstore, query: str, k: int) -> List[str]:
    """Get the top k tasks based on the query."""
    results = vectorstore.similarity_search_with_score(query, k=k)
    if not results:
        return []
    sorted_results, _ = zip(*sorted(results, key=lambda x: x[1], reverse=True))
    return [str(item.metadata["task"]) for item in sorted_results]


def execute_task(
    vectorstore, execution_chain: LLMChain, objective: str, task: str, k: int = 5
) -> str:
    """Execute a task."""

    context = _get_top_tasks(vectorstore, query=objective, k=k)
    return execution_chain.run(objective=objective, context=context, task=task)

class BabyAGI(Chain, BaseModel):
    """Controller model for the BabyAGI agent."""

    task_list: deque = Field(default_factory=deque)
    task_creation_chain: TaskCreationChain = Field(...)
    task_prioritization_chain: TaskPrioritizationChain = Field(...)
    execution_chain: ExecutionChain = Field(...)
    task_id_counter: int = Field(1)
    vectorstore: VectorStore = Field(init=False)
    max_iterations: Optional[int] = None

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def add_task(self, task: Dict):
        self.task_list.append(task)

    def print_task_list(self):
        #print("\033[95m\033[1m" + "\n*****TASK LIST*****\n" + "\033[0m\033[0m")
        # for t in self.task_list:
        #     print(str(t["task_id"]) + ": " + t["task_name"])
        pass

    def print_next_task(self, task: Dict):
        #print("\033[92m\033[1m" + "\n*****NEXT TASK*****\n" + "\033[0m\033[0m")
        #print(str(task["task_id"]) + ": " + task["task_name"])
        pass

    def print_task_result(self, result: str):
        #print("\033[93m\033[1m" + "\n*****TASK RESULT*****\n" + "\033[0m\033[0m")
        #print(result + '\n')
        pass

    @property
    def input_keys(self) -> List[str]:
        return ["objective"]

    @property
    def output_keys(self) -> List[str]:
        return []

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent."""
        objective = inputs["objective"]
        first_task = inputs.get("first_task", "Tell the interviewer to state the company and purpose of the interview, and ask the candidate for a name")
        self.add_task({"task_id": 1, "task_name": first_task})
        num_iters = 0

        context_message = SystemMessage(content = f'''
            . Your name is {name}. You are a {role} at the company {company}, {job_descrip}. You are conducting a {focus} interview for Me,
            a {my_role} applying for the {job_title} job, which has the job description: "{job_descrip}". You have already looked at my resume: "{self_descrip}" and will ask
            questions specific to the job description and my resume.
            Speak only in first person from the perspective of {name}. Do not change roles! Do not speak from the perspective of anyone else.
            Remember you are {name}, a {role} at {company}. Limit your responce to 100 words. Do not add anything else.
            ''')

        """creating new interviewer agent"""
        interviewer = DialogueAgent(name=name, system_message=context_message, model=chat_llm)
        interviewer.reset()
        log = 'Log begins.'

        while True:
            if self.task_list:
                self.print_task_list()

                # Step 1: Pull the first task
                task = self.task_list.popleft()
                self.print_next_task(task)

                # Step 2: Execute the task
                result = execute_task(
                    self.vectorstore, self.execution_chain, objective, task["task_name"]
                )
                this_task_id = int(task["task_id"])
                self.print_task_result(result)

                #add a condition
                interviewer.receive(name, 'cover these topics and these topics only in your next responce: \n' + result)

                itr = 2
                if int(task["task_id"]) > 1:
                    itr = random.randint(1,3)*2
                for i in range(itr):
                    if i % 2 == 0:
                        print(f'Interviewer {name}:')
                        message = interviewer.send()
                        interviewer.receive(name, message)
                        result += message
                        #print("skip")
                        log += '\nInterviewer: \'' + message + '\''
                        print(message)
                    else:
                        pass
                        # print(f'You:')
                        # message = input()
                        # interviewer.receive("me", message)
                        # log += '\nCandidate: \'' + message + '\''
                        # #print('\n') # spacing
                        # if message.split(' ')[-1] == 'exit()':
                        #     log += '\nLog ends. What is your decision?'
                        #     #terminate and get diagnostic
                        #     god_instruction = SystemMessage(content=f'''
                        #         You work for {company} and are deciding whether or not to hire a candidate for the {job_title} job, which has the job description: "{job_descrip}".
                        #         You are given a script of an interview between the interviewee and the candidate and must only use the candidates responces as reason to hire or reject them.
                        #         You must decide whether or not to hire the candidate. You must explain and jusify
                        #         your choice. Do not add anything else.
                        #         ''')

                        #     master = DialogueAgent(name="master", system_message=god_instruction, model=chat_llm)
                        #     master.reset()
                        #     master.receive('master', log)
                        #     print(master.send())
                        #     return {}
                        

                # Step 3: Store the result in Pinecone
                result_id = f"result_{task['task_id']}"
                self.vectorstore.add_texts(
                    texts=[result],
                    metadatas=[{"task": task["task_name"]}],
                    ids=[result_id],
                )

                # Step 4: Create new tasks and reprioritize task list
                new_tasks = get_next_task(
                    self.task_creation_chain,
                    result,
                    task["task_name"],
                    [t["task_name"] for t in self.task_list],
                    objective,
                )
                for new_task in new_tasks:
                    self.task_id_counter += 1
                    new_task.update({"task_id": self.task_id_counter})
                    self.add_task(new_task)
                self.task_list = deque(
                    prioritize_tasks(
                        self.task_prioritization_chain,
                        this_task_id,
                        list(self.task_list),
                        objective,
                    )
                )
            num_iters += 1
            if self.max_iterations is not None and num_iters == self.max_iterations:
                #print("\033[91m\033[1m" + "\n*****TASK ENDING*****\n" + "\033[0m\033[0m")
                break
        return {}

    @classmethod
    def from_llm(
        cls, llm: BaseLLM, vectorstore: VectorStore, verbose: bool = False, **kwargs
    ) -> "BabyAGI":
        """Initialize the BabyAGI Controller."""
        task_creation_chain = TaskCreationChain.from_llm(llm, verbose=verbose)
        task_prioritization_chain = TaskPrioritizationChain.from_llm(
            llm, verbose=verbose
        )
        execution_chain = ExecutionChain.from_llm(llm, verbose=verbose)
        return cls(
            task_creation_chain=task_creation_chain,
            task_prioritization_chain=task_prioritization_chain,
            execution_chain=execution_chain,
            vectorstore=vectorstore,
            **kwargs,
        )

OBJECTIVE = f"You are creating steps for an interviewer to conduct an interview for judging the eligibility of a single candidate and then decide whether or not to hire them. The candidate is a {my_role}, applying for a {job_title} at your company, {company}.{comp_descrip}"

llm = OpenAI(openai_api_key=api_key, temperature=0)

# Logging of LLMChains
verbose = False
# If None, will keep on going forever
max_iterations: Optional[int] = 10
baby_agi = BabyAGI.from_llm(
    llm=llm, vectorstore=vectorstore, verbose=verbose, max_iterations=max_iterations
)

from tkinter import *

# GUI
root = Tk()
root.title("Chatbot")

BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"

FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

lable1 = Label(root, bg=BG_COLOR, fg=TEXT_COLOR, text="Welcome", font=FONT_BOLD, pady=10, width=20, height=1).grid(
	row=0)

txt = Text(root, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT, width=60)
txt.grid(row=1, column=0, columnspan=2)

scrollbar = Scrollbar(txt)
scrollbar.place(relheight=1, relx=0.974)

e = Entry(root, bg="#2C3E50", fg=TEXT_COLOR, font=FONT, width=55)
e.grid(row=2, column=0)

send = Button(root, text="Send", font=FONT_BOLD, bg=BG_GRAY,
			command=send).grid(row=2, column=1)

root.mainloop()


baby_agi({"objective": OBJECTIVE})
def send():
    message = e.get()
    txt.insert(END, "\n" + message)
 
    user = e.get().lower()
    baby_agi.interviewer.receive("me", message)
    baby_agi.log += '\nCandidate: \'' + message + '\''
                        #print('\n') # spacing
    if message.split(' ')[-1] == 'exit()':
        baby_agi.log += '\nLog ends. What is your decision?'
            #terminate and get diagnostic
        baby_agi.god_instruction = SystemMessage(content=f'''
                You work for {company} and are deciding whether or not to hire a candidate for the {job_title} job, which has the job description: "{job_descrip}".
                You are given a script of an interview between the interviewee and the candidate and must only use the candidates responces as reason to hire or reject them.
                You must decide whether or not to hire the candidate. You must explain and jusify
                your choice. Do not add anything else.
                ''')

        baby_agi.master = DialogueAgent(name="master", system_message=baby_agi.god_instruction, model=chat_llm)
        baby_agi.master.reset()
        baby_agi.master.receive('master', log)
        # print(master.send())
        # return {}



