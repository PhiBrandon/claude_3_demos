import boto3
import os
from dotenv import load_dotenv


class AgentsBedrock:
    def __init__(self, profile_name: str = None):
        """Initialize the AgentsBedrock class."""
        load_dotenv()
        profile_name = profile_name or os.getenv("AWS_PROFILE")
        self.session = boto3.Session(profile_name=profile_name)
        self.client = self.session.client("bedrock-agent-runtime")

    def _create_session(self):
        """Create a new AWS session."""
        return boto3.Session(profile_name=self.profile_name)

    def _invoke_agent(self, agent_id: str, body: dict, session_id: str):
        """Invoke the agent."""
        return self.client.invoke_agent(agentId=agent_id, sessionId=session_id , inputText=body, enableTrace=True, )
