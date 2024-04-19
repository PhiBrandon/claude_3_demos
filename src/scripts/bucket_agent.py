import boto3
from typing import List
from llama_index.core.agent import ReActAgent, Task
from llama_index.core.tools import FunctionTool
from llama_index.llms.anthropic import Anthropic
from dotenv import load_dotenv
from llama_index.llms.mistralai import MistralAI
from llama_index.llms.together import TogetherLLM
from llama_index.llms.fireworks import Fireworks
from llama_index.llms.litellm import LiteLLM
import os
from pydantic import BaseModel
import langfuse
import litellm


load_dotenv()
litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]


AWS_PROFILE = os.getenv("AWS_PROFILE")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
# claude-3-sonnet-20240229
# claude-3-opus-20240229
# claude-3-haiku-20240307
llm = Anthropic(model="claude-3-haiku-20240307")
# llm = Fireworks(model="accounts/fireworks/models/hermes-2-pro-mistral-7b", api_key=FIREWORKS_API_KEY)
# llm = MistralAI(model="mistral-large")
# NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO
# NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT
# Qwen/Qwen1.5-72B-Chat
# deepseek-ai/deepseek-coder-33b-instruct
# llm = TogetherLLM(model="Qwen/Qwen1.5-72B-Chat", api_key=TOGETHER_API_KEY)
""" llm = LiteLLM(
    model="together_ai/meta-llama/Llama-3-70b-chat-hf",
) """
llm.complete("what's up")
#duckduck = download_tool("DuckDuckGoSearchToolSpec")
#tool_ducks = duckduck().to_tool_list()


def get_buckets():
    """List all S3 buckets in an AWS Account"""
    s3 = boto3.client("s3")
    response = s3.list_buckets()
    return response


def get_bucket_objects(bucket_name: str):
    """List all objects in an S3 bucket"""
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=bucket_name)
    return response

def write_to_file(file_name: str ,text: str):
    with open(file_name, "w") as f:
        f.write(text)

def read_file(file_name: str):
    """Read a file and return the content as a string"""
    with open(file_name, "r") as f:
        return f.read()

write_tool = FunctionTool.from_defaults(fn=write_to_file)
bucket_tool = FunctionTool.from_defaults(fn=get_buckets)
bucket_object_tool = FunctionTool.from_defaults(fn=get_bucket_objects)
read_file_tool = FunctionTool.from_defaults(fn=read_file)
tools = [bucket_tool, write_tool, bucket_object_tool, read_file_tool]
# tools = tool_ducks

agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, max_iterations=50)


class OutputList(BaseModel):
    output: List[str]


output = agent.chat(f"List the  s3 buckets that start with data and the largest 3 files between them. Write the file names, file size in MB(it will be given to you in bytes), and the day they were created, make it look nice, to a file name niceoutput.txt")
print(output)

""" tasks: List[Task] = agent.list_tasks()
print(len(tasks))
task: Task = tasks[-1].task
print(task.dict().keys())
print(task.extra_state.keys())
sources = task.extra_state["sources"]
for s in sources:
    print(s) """
