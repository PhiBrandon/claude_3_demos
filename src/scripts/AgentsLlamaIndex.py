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

# Let's move up to the top layer
# We can use the ReActAgent as the Agent Runner and Worker combined, the high level API
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)
task = agent.create_task("Multiply 19 and 2")
# Let's run the step for the particular task
output = agent.run_step(task.task_id)
# Print the output
print(output)

task_list = agent.list_tasks()
for task in task_list:
    print(task.task)
# We see that we only have one task, let's create a more complex task
    
complex_task = agent.create_task("Multiply 10 and 2. Add 10 to the result. Then multiply that result by 2.")
complex_output = agent.run_step(complex_task.task_id)
print(complex_output)

# Let's get the list of task now
task_list_updated = agent.list_tasks()
for t in task_list_updated:
    print(t.task)
    print(t.task.extra_state['current_reasoning'])
    print("="*20)
# Now we see both the previous task and the new task


detail_test_trace = lanfuse.trace(name="agent_detail_extraction", metadata={"model": "claude-3-sonnet-20240229"})
def get_detail_json(text: str):
    class UserDetail(BaseModel):
        name: str
        age: int
    system_append = f"Here's a JSON schema to follow:{UserDetail.model_json_schema()} Output a valid JSON object but do not repeat the schema."
    user_prompt = f"Please extract the user detail from the text: {text}"
    complete = system_append + "\n" + user_prompt
    output = llm.complete(complete)
    return output

user_detail_tool = FunctionTool.from_defaults(fn=get_detail_json)
tools = [add_tool, multiply_tool, user_detail_tool]
detail_agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)
task = detail_agent. create_task("Extract the user detail into json: Duyen is 22 years old.")
generation = detail_test_trace.generation(input={"task_input": "Extract the user detail into json: Duyen is 22 years old.", "task": task} )
detail_output = detail_agent.run_step(task.task_id)
generation.end(output={"output": detail_output.output.response, "reasoning": task.extra_state['current_reasoning']})
print(detail_output)
