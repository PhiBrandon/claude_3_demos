from llama_index.llms.anthropic import Anthropic
from llama_index.core import Settings
from langfuse import Langfuse
from dotenv import load_dotenv
from llama_index.core.agent import AgentRunner, ReActAgentWorker, ReActAgent
from llama_index.core.tools import FunctionTool
import nest_asyncio
from langfuse.client import StatefulTraceClient
from pydantic import BaseModel

nest_asyncio.apply()

load_dotenv()

# Instantiate the Langfuse client
lanfuse = Langfuse()
llm = Anthropic(model="claude-3-sonnet-20240229")

# Define a function that the LLM can use
def add(a: int, b: int):
    return a + b

# The Tools that 
add_tool = FunctionTool.from_defaults(fn=add)
tools = [add_tool]

agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)
trace: StatefulTraceClient = lanfuse.trace(
    name="agent_add_test", metadata={"model": "claude-3-sonnet-20240229"}
)
gen = trace.generation(
    id="agent_add_test", input={"query": "What is the sum of 2 and 3?"}
)
output = agent.chat("What is the sum of 2 and 3?")
gen.end(output={"output": output})

print(output)

# Let's try something that isn't boilerplate
text = """

This is some redacted text so that we are able to not commit things that are sensative to the internet and my cool
people that i interact with on the daily huehuehuehuehuehuehuehuehue.
"""


def get_summary(this_text: str) -> str:
    this_output = llm.complete(
        f"You are a mystical wizard. Summarize the text: {this_text}"
    )
    return this_output.text


summary_tool = FunctionTool.from_defaults(fn=get_summary)

def multiply(a: int, b: int):
    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)

tools = [add_tool, multiply_tool]

math_agent = ReActAgentWorker.from_tools(tools, llm=llm, verbose=True)
agent = AgentRunner(math_agent)
trace: StatefulTraceClient = lanfuse.trace(
    name="agent_math_text_v01", metadata={"model": "claude-3-sonnet-20240229"}
)

class Maths(BaseModel):
    input: str
    answer: float


task_input = f"Add 100 + 2. Multiply the result with 3. Here's a JSON schema to follow:{Maths.model_json_schema()} Output a valid JSON object but do not repeat the schema."
task = agent.create_task(task_input)
gen = trace.generation(input={"query": task_input})
step_output = agent.run_step(task.task_id)
tasks = agent.list_tasks()
current_task = tasks[-1]
reason = tasks[-1].task.extra_state['current_reasoning'][0].thought
print(tasks[-1].task.extra_state['current_reasoning'][0].thought)
gen.end(output={"output": step_output.output.response, "tasks": str(current_task), "reasoning": reason})

print(step_output)
