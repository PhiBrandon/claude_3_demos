import boto3

# function that get's s3 buckets
def getBuckets():
    s3 = boto3.client("s3")
    response = s3.list_buckets()
    return response['Buckets']

def lambda_handler(event, context):
    result = None

    if event["apiPath"] == "/getBuckets":
        result = getBuckets()

    if result:
        print("Answer:", result)

    else:
        result = "Getting Buckets Failed."

    response_body = {"application/json": {"body": str(result)}}

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
