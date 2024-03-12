import boto3
from typing import List
from llama_index.core.agent import ReActAgent, Task
from llama_index.core.tools import FunctionTool
from llama_index.llms.anthropic import Anthropic
from dotenv import load_dotenv
from llama_index.core.tools.download import download_tool
from llama_index.llms.mistralai import MistralAI
from llama_index.llms.together import TogetherLLM
import os

load_dotenv()

AWS_PROFILE = os.getenv("AWS_PROFILE")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
# claude-3-sonnet-20240229
# claude-3-opus-20240229
llm = Anthropic(model="claude-3-opus-20240229")
#llm = MistralAI(model="mistral-large")
# NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO
# NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT
# Qwen/Qwen1.5-72B-Chat
# deepseek-ai/deepseek-coder-33b-instruct
#llm = TogetherLLM(model="Qwen/Qwen1.5-72B-Chat", api_key=TOGETHER_API_KEY)
llm.complete("what's up")
duckduck = download_tool("DuckDuckGoSearchToolSpec")
tool_ducks = duckduck().to_tool_list()


def list_s3_buckets():
    s3 = boto3.Session(profile_name=AWS_PROFILE).client("s3")
    bucket_list = s3.list_buckets()
    return bucket_list["Buckets"]

def write_to_file(text: str):
    with open("bucket_list.txt", "w") as f:
        f.write(text)

write_tool = FunctionTool.from_defaults(fn=write_to_file)
bucket_tool = FunctionTool.from_defaults(fn=list_s3_buckets)
tools = [bucket_tool, write_tool] + tool_ducks

agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)
output = agent.chat(f"List my s3 buckets, search for updates on doge, then write all of this to a file.")

tasks : List[Task] = agent.list_tasks()
print(len(tasks))
task : Task = tasks[-1].task
print(task.dict().keys())
print(task.extra_state.keys())
sources = task.extra_state['sources']
for s in sources:
    print(s)
