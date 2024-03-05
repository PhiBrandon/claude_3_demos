import boto3
import json
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.client import StatefulTraceClient, StatefulGenerationClient


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


class TraceInformation(BaseModel):
    model: str
    model_parameters: ModelParameters
    input: List[dict]
    metadata: Optional[dict] = None


class SyntheticUserDetails(BaseModel):
    """Minimum of 20 new examples. Do not repeat previously given examples"""

    data_examples: List[UserDetail]


class SyntheticUserSentences(BaseModel):
    """Minimum of 20 new examples. Do not repeat previously given examples"""

    data_examples: List[str]


class BedrockClient:
    def __init__(self, profile_name):
        load_dotenv()
        session = boto3.Session(profile_name=profile_name)
        self.bedrock_rt = session.client("bedrock-runtime")
        self.langfuse = Langfuse()

    def create_trace(self, name: str, metadata: dict) -> StatefulTraceClient:
        return self.langfuse.trace(id=name, name=name, metadata=metadata)

    def create_generation(
        self, trace: StatefulTraceClient, trace_information: TraceInformation, name: str
    ) -> StatefulGenerationClient:
        return trace.generation(
            name=name,
            model=trace_information.model,
            model_parameters=trace_information.model_parameters,
            input=trace_information.input,
            metadata=trace_information.metadata,
        )

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
    client = BedrockClient(profile_name="bigfrog")

    data = [
        "Brandon is 30 and a data engineer.",
        "Jason is 22 and a GOAT.",
        "Duyen is 25 and a Product Manager.",
        "John is 99 and a retired astronaut.",
        "Jeff is 45, he drives cars for a living.",
        "Sara is 33 and spends all of her time at home raising children.",
    ]

    dataset_name = "UserDetail"
    for d in data:
        prompt = [
            {
                "role": "user",
                "content": f"Extract the user detail from the following text: {d}",
            }
        ]
        system = f"Here's a JSON schema to follow: {UserDetail.model_json_schema()} Output a valid JSON object but do not repeat the schema."
        model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
        user_trace = client.create_trace(
            "claude_3_user_detail_v0.07",
            {"model": "anthropic.claude-3-sonnet-20240229-v1:0"},
        )
        trace_info = TraceInformation(
            model=model_id,
            model_parameters=ModelParameters(
                max_tokens=3000, temperature=0, system=system
            ),
            input=prompt,
            metadata={
                "system_prompt": system,
                "pydantic_schema": UserDetail.model_json_schema(),
            },
        )
        gen = client.create_generation(user_trace, trace_info, "user_detail")
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
            gen.end(
                output={
                    "output": text_resp["content"][0]["text"],
                }
            )
            client.langfuse.create_dataset_item(
                dataset_name, input=d, expected_output=valid_user
            )
        except Exception as e:
            gen.end(
                output={
                    "output": text_resp["content"][0]["text"],
                }
            )

    def process_observations(client, trace_name) -> List:
        url = client.langfuse.get_trace(trace_name)
        examples = []
        observations = url.observations
        for obv in observations:
            if obv.type == "GENERATION":
                if obv.output:
                    print("Input:")
                    print(obv.input[0]["content"])
                    print()
                    print("System:")
                    print(obv.metadata["system_prompt"])
                    print()

                    print("Output:")
                    output = UserDetail.model_validate_json(obv.output["output"])
                    examples.append(output)
                    print(output.name)
                    print(output.age)
                    print(output.occupation)
                    print("\n\n")
        return examples

    # Call the function
    examples = process_observations(client, "claude_3_user_detail_v0.05")

    def generate_additional_examples(client, examples):
        prompt = [
            {
                "role": "user",
                "content": f"Generate addition examples based on the following examples: {str(examples)}",
            }
        ]

        system = f"Here's a JSON schema to follow: {SyntheticUserDetails.model_json_schema()} Output a valid JSON object but do not repeat the schema."
        synthetic_trace = client.create_trace(
            "claude_3_synthetic_user_detail_v0.02",
            {"model": "anthropic.claude-3-sonnet-20240229-v1:0"},
        )
        model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
        trace_info = TraceInformation(
            model=model_id,
            model_parameters=ModelParameters(max_tokens=3000, temperature=0, system=system),
            input=prompt,
            metadata={
                "system_prompt": system,
                "pydantic_schema": SyntheticUserDetails.model_json_schema(),
            },
        )
        gen = client.create_generation(synthetic_trace, trace_info, "synthetic_user_detail")
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
        new_examples = SyntheticUserDetails.model_validate_json(
            text_resp["content"][0]["text"]
        )
        examples.extend(new_examples.data_examples)
        return examples

    def generate_additional_sentences(client, data, example_sentences):
        example_sentences = []
        prompt = [
            {
                "role": "user",
                "content": f"Generate addition examples based on the following examples: {str(data)} {str(example_sentences)}",
            }
        ]
        system = f"Here's a JSON schema to follow: {SyntheticUserSentences.model_json_schema()} Output a valid JSON object but do not repeat the schema."
        synthetic_trace = client.create_trace(
            "claude_3_synthetic_user_sentences_v0.03",
            {"model": "anthropic.claude-3-sonnet-20240229-v1:0"},
        )
        model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
        trace_info = TraceInformation(
            model=model_id,
            model_parameters=ModelParameters(max_tokens=3000, temperature=0, system=system),
            input=prompt,
            metadata={
                "system_prompt": system,
                "pydantic_schema": SyntheticUserSentences.model_json_schema(),
            },
        )
        gen = client.create_generation(
            synthetic_trace, trace_info, "synthetic_user_sentences"
        )
        text_resp = client.get_claude_response(
            messages=prompt,
            system=system,
            token_count=3000,
            temp=0.5,
            topP=1,
            topK=0,
            stop_sequence=["Human:"],
            model_id=model_id,
        )
        new_sentences = SyntheticUserSentences.model_validate_json(
            text_resp["content"][0]["text"]
        )

        data.extend(example_sentences)
        return data


if __name__ == "__main__":
    main()
