import boto3
from typing import List
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.anthropic import Anthropic
from dotenv import load_dotenv
from llama_index.core.tools.download import download_tool
from llama_index.llms.mistralai import MistralAI
from llama_index.llms.together import TogetherLLM

load_dotenv()

# llm = Anthropic(model="claude-3-opus-20240229")
llm = MistralAI(model="mistral-large")
llm = TogetherLLM(
    model="Qwen/Qwen1.5-72B-Chat", api_key="together_ai_api_key"
)
llm.complete("what's up")
duckduck = download_tool("DuckDuckGoSearchToolSpec")
tool_ducks = duckduck().to_tool_list()


def list_s3_buckets():
    s3 = boto3.Session(profile_name="s3_local_profil").client("s3")
    bucket_list = s3.list_buckets()
    return bucket_list["Buckets"]


bucket_tool = FunctionTool.from_defaults(fn=list_s3_buckets)
tools = [bucket_tool] + tool_ducks
print(tools)

agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)
output = agent.chat(f"What's the latest mistral large ai news?")
print(output)
