import boto3
import json
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv


class UserDetail(BaseModel):
    """
    Details about the user in the string.
    """

    name: str
    age: int
    occupation: str


class ModelParameters(BaseModel):
    max_tokens: int
    temperature: float
    system: Optional[str] = None


class BedrockClient:
    def __init__(self, profile_name):
        load_dotenv()
        session = boto3.Session(profile_name=profile_name)
        self.bedrock_rt = session.client("bedrock-runtime")

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
        messages="",
        system="",
        token_count=250,
        temp=0,
        topP=1,
        topK=250,
        stop_sequence=["Human:"],
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    ):
        body = self.create_claude_body(
            messages=messages,
            system=system,
            token_count=token_count,
            temp=temp,
            topP=topP,
            topK=topK,
            stop_sequence=stop_sequence,
        )
        response = self.bedrock_rt.invoke_model(modelId=model_id, body=json.dumps(body))
        response = json.loads(response["body"].read().decode("utf-8"))
        return response


def main():
    client = BedrockClient(profile_name="your-aws-profile-name-here")

    data = [
        "Brandon is 30 and a data engineer.",
        "Jason is 22 and a GOAT.",
        "Duyen is 25 and a Product Manager.",
        "John is 99 and a retired astronaut.",
        "Jeff is 45, he drives cars for a living.",
        "Sara is 33 and spends all of her time at home raising children.",
    ]

    for d in data:
        prompt = [
            {
                "role": "user",
                "content": f"Extract the user detail from the following text: {d}",
            }
        ]
        system = f"Here's a JSON schema to follow: {UserDetail.model_json_schema()} Output a valid JSON object but do not repeat the schema."
        model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
        text_resp = client.get_claude_response(
            messages=prompt,
            system=system,
            token_count=3000,
            temp=0,
            topP=1,
            topK=0,
            stop_sequence=["Human:"],
            model_id=model_id,
        )

        try:
            valid_user = UserDetail.model_validate_json(text_resp["content"][0]["text"])
            print(valid_user)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
