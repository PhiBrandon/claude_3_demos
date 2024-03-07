import boto3
import os
from dotenv import load_dotenv
import logging
import json
from langfuse import Langfuse

load_dotenv()
langfuse = Langfuse()
page_text = open("data/page.txt", "r").read()
client = boto3.client("bedrock-agent-runtime")


def invoke_agent(langfuse, page_text, client):
    output = client.invoke_agent(
        inputText=f"Summarize the text: {page_text}",
        agentId="5BCPJZ7OIG",
        agentAliasId="EJH2X2JTGM",
        sessionId="test",
        enableTrace=True,
        endSession=False,
    )
    event_stream = output["completion"]
    traces = []
    agent_answer = ""
    try:
        trace = langfuse.trace(
            name="agent_test_v01", metadata={"model": "bedrock-agent-runtime"}
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


agent_answer = invoke_agent(langfuse, page_text, client)

print(agent_answer)
