# Claude and Langfuse Spaghetti

## Folder Contents
- data - This folder contains data from https://github.com/FranxYao/chain-of-thought-hub
- src - This contains two folders, Notebooks and Scripts. 
- lambda_functions - This folder contains the lambda function and OpenAPI spec that are utilized by Bedrock to do function calling, when needed.

You'll find the playground where i've started playing around with the Bedrock implementation of Claude 3 Opus/Sonnet. Across the notebooks and scripts, you'll find simple implementations of a BedrockRuntime client
that will allow you to invoke the Claude 3 models.

## Setup
1. Create .env file
2. Add ANTHROPIC_API_KEY that you've claimed from the Anthropic console. https://console.anthropic.com/dashboard
3. Make sure you have you AWS keys in your environment, you can add them to the env, but also you can simply export them via the console. https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html

*Note*
src/notebooks/bedrock.ipynb is the base implementation that all other implementations follow.
src/notebooks/llama_anthropic.ipynb is testing of the Llama_index implementation of the Anthropic client for Claude 3.
src/notebooks/evals.ipynb is running evaluations with the Anthropic client.
anthropic-tools/* is a folder from https://github.com/anthropics/anthropic-tools/tree/main/tool_use_package . If you follow the README, you will be able to create your own functions that even Claude Haiku
can run successfully.
anthropic-tools/tooler.py is an example of using tools. You're able to list S3 buckets, get the time, then write it to file locally.


If you have questions feel free to reach out.
