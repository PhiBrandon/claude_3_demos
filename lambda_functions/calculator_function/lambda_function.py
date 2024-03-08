
# Function that accepts a youtube link
def add(num1, num2):
    return num1 + num2

def lambda_handler(event, context):
    result = None

    if event["apiPath"] == "/addNumbers":
        num_1 = event["requestBody"]["content"]["application/json"]["properties"][0][
            "value"
        ]
        num_2 = event["requestBody"]["content"]["application/json"]["properties"][1][
            "value"
        ]
        result = add(num_1, num_2)

    if result:
        print("Answer:", result)

    else:
        result = "Add Failed."

    response_body = {"application/json": {"body": result}}

    action_response = {
        "actionGroup": event["actionGroup"],
        "apiPath": event["apiPath"],
        "httpMethod": event["httpMethod"],
        "httpStatusCode": 200,
        "responseBody": response_body,
    }

    session_attributes = event["sessionAttributes"]
    prompt_session_attributes = event["promptSessionAttributes"]

    api_response = {
        "messageVersion": "1.0",
        "response": action_response,
        "sessionAttributes": session_attributes,
        "promptSessionAttributes": prompt_session_attributes,
    }

    return api_response
