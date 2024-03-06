# This code was refactored based on https://claude.ai/chat/6cd88338-edea-4d61-91c6-06c135496b02.
"""
Step 1:
Given some code, you are to analyze the code and give feedback. Your analysis should account for readability, reusability, and efficiency of the code presented. You should suggest recommendations to improve each of the metrics presented here.

Step 2:
Modify the code to implement the recommendations mentioned here.
"""
import boto3
import json
from dotenv import load_dotenv
import os

class BedrockClient:
    """
    A client for interacting with the Bedrock service.

    Args:
        profile_name (str): The name of the AWS profile to use for authentication.

    Attributes:
        client (botocore.client.BaseClient): The Bedrock runtime client.
    """

    def __init__(self, profile_name=None):
        load_dotenv()
        profile_name = profile_name or os.getenv("AWS_PROFILE")
        self.session = boto3.Session(profile_name=profile_name)
        self.client = self.session.client("bedrock-runtime")

    def _create_session(self):
        """Create a new AWS session."""
        return boto3.Session(profile_name=self.profile_name)

    def create_claude_body(
        self,
        messages=[{"role": "user", "content": "Hello!"}],
        system="",
        token_count=150,
        temp=0,
        topP=1,
        topK=250,
        stop_sequence=["Human"],
    ):
        """
        Create the request body for the Claude model.

        Args:
            messages (list, optional): List of messages in the conversation. Each message should have a "role" and "content" field. Defaults to [{"role": "user", "content": "Hello!"}].
            system (str, optional): The system message. Defaults to "".
            token_count (int, optional): The maximum number of tokens in the response. Defaults to 150.
            temp (int, optional): The temperature for generating the response. Defaults to 0.
            topP (int, optional): The top-p value for generating the response. Defaults to 1.
            topK (int, optional): The top-k value for generating the response. Defaults to 250.
            stop_sequence (list, optional): List of stop sequences to end the response. Defaults to ["Human"].

        Returns:
            dict: The request body for the Claude model.
        """
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": messages,
            "max_tokens": token_count,
            "temperature": temp,
            "anthropic_version": "",
            "top_k": topK,
            "top_p": topP,
            "stop_sequences": stop_sequence,
            "system": system,
        }
        return body

    def get_claude_response(
        self,
        messages=[],
        system="",
        token_count=250,
        temp=0,
        topP=1,
        topK=250,
        stop_sequence=["Human:"],
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    ):
        """
        Get the response from the Claude model.

        Args:
            messages (list, optional): List of messages in the conversation. Each message should have a "role" and "content" field. Defaults to [].
            system (str, optional): The system message. Defaults to "".
            token_count (int, optional): The maximum number of tokens in the response. Defaults to 250.
            temp (int, optional): The temperature for generating the response. Defaults to 0.
            topP (int, optional): The top-p value for generating the response. Defaults to 1.
            topK (int, optional): The top-k value for generating the response. Defaults to 250.
            stop_sequence (list, optional): List of stop sequences to end the response. Defaults to ["Human:"].
            model_id (str, optional): The ID of the Claude model. Defaults to "anthropic.claude-3-sonnet-20240229-v1:0".

        Returns:
            dict: The response from the Claude model.
        """
        body = self.create_claude_body(
            messages=messages,
            system=system,
            token_count=token_count,
            temp=temp,
            topP=topP,
            topK=topK,
            stop_sequence=stop_sequence,
        )
        response = self.client.invoke_model(modelId=model_id, body=json.dumps(body))
        response = json.loads(response["body"].read().decode("utf-8"))
        return response