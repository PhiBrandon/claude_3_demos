import boto3
import os
from dotenv import load_dotenv
import json
from langfuse import Langfuse

load_dotenv()
langfuse = Langfuse()
client = boto3.client("bedrock-agent-runtime")
AGENT_ID = os.getenv("AGENT_ID")
AGENT_ALIAS_ID = os.getenv("AGENT_ALIAS_ID")


def invoke_agent(langfuse, client):
    output = client.invoke_agent(
        inputText=f"Get the titles from this youtube channel: https://www.youtube.com/@JaysonCasper",
        agentId=AGENT_ID,
        agentAliasId=AGENT_ALIAS_ID,
        sessionId="test",
        enableTrace=True,
        endSession=False,
    )
    event_stream = output["completion"]
    traces = []
    agent_answer = ""
    try:
        trace = langfuse.trace(
            name="agent_test_v02", metadata={"model": "bedrock-agent-runtime"}
        )
        for event in event_stream:
            if "chunk" in event:
                data = event["chunk"]["bytes"]
                agent_answer = data.decode("utf8")
                end_event_received = True
                gen = trace.generation(name="agent_test_answer")
                gen.end(output={"output": agent_answer})
            # End event indicates that the request finished successfully
            elif "trace" in event:
                gen = trace.generation(name="agent_test_trace")
                traces.append(json.dumps(event["trace"]))
                gen.end(output={"output": json.dumps(event["trace"])})
            else:
                raise Exception("unexpected event.", event)
    except Exception as e:
        raise Exception("unexpected event.", e)
    return agent_answer


agent_answer = invoke_agent(langfuse, client)

print(agent_answer)
