import boto3
import os
from dotenv import load_dotenv
import json
from langfuse import Langfuse
import datetime

load_dotenv()
langfuse = Langfuse()
client = boto3.client("bedrock-agent-runtime")
AGENT_ID = os.getenv("AGENT_ID")
AGENT_ALIAS_ID = os.getenv("AGENT_ALIAS_ID")
print(AGENT_ALIAS_ID)


# Get the titles from this youtube channel: https://www.youtube.com/@QuinnNolan
def invoke_agent(langfuse, client):
    output = client.invoke_agent(
        inputText=f"Get the video titles from https://www.youtube.com/@DiamondDave1",
        agentId=AGENT_ID,
        agentAliasId=AGENT_ALIAS_ID,
        sessionId="test_new_agent_v04",
        enableTrace=True,
        endSession=False,
    )
    event_stream = output["completion"]
    traces = []
    agent_answer = ""
    try:
        trace = langfuse.trace(
            name="agent_add_youtube_test_v04",
            metadata={"model": "bedrock-agent-runtime"},
        )
        for event in event_stream:
            gen = trace.generation(
                name="agent_generation"
            )
            if "chunk" in event:
                data = event["chunk"]["bytes"]
                agent_answer = data.decode("utf8")
                end_event_received = True
                gen.update(name="agent_answer")
                gen.end(
                    output={
                        "output": agent_answer,
                    },
                    
                )
            # End event indicates that the request finished successfully
            elif "trace" in event:
                gen.update(name="agent_trace")
                traces.append(json.dumps(event["trace"]))
                gen.end(
                    output={"output": json.dumps(event["trace"])},
                )
            else:
                raise Exception("unexpected event.", event)
    except Exception as e:
        raise Exception("unexpected event.", e)
    return agent_answer


agent_answer = invoke_agent(langfuse, client)

print(agent_answer)
