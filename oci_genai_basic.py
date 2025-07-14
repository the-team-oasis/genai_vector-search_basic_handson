# coding: utf-8
##########################################################################
# Copyright (c) 2023, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0
# as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at
# https://www.apache.org/licenses/LICENSE-2.0. You may choose either license.
##########################################################################
# Reference Link: https://python.langchain.com/docs/integrations/llms/oci_generative_ai/
##########################################################################
# Install packages
# pip install -U oci==2.141.1 langchain_core==0.3.29 langchain-community==0.3.14
##########################################################################
# Application Command line
# python oci_genai_basic.py --message "Hello"
# python oci_genai_basic.py -m "Ask a question"
# python oci_genai_basic.py -m "Hello" -c my_config.json
##########################################################################
from langchain_community.chat_models import ChatOCIGenAI
from langchain_core.messages import HumanMessage
import argparse
import json

# Command line argument parsing
parser = argparse.ArgumentParser(description='Chat with OCI Generative AI Service using LangChain.')
parser.add_argument('-m', '--message', 
                   type=str, 
                   required=True,
                   help='Message to send to LLM')
parser.add_argument('-c', '--config', 
                   type=str, 
                   default='llm_parameter_config.json',
                   help='Path to LLM parameter config file')

args = parser.parse_args()

# Load LLM parameters from config file
try:
    with open(args.config, 'r') as f:
        llm_params = json.load(f)
    print(f"Loaded LLM parameters from {args.config}")
except FileNotFoundError:
    print(f"Config file {args.config} not found. Using default parameters.")
    llm_params = {
        "max_tokens": 600,
        "temperature": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "top_p": 0.75,
        "top_k": 0
    }

# Setup basic variables
# Auth Config
# TODO: Please update config profile name and use the compartmentId that has policies grant permissions for using Generative AI Service
COMPARTMENT_ID = "ocid1.compartment.oc1..aaaaaaaacoh3hxxu7cerb7zwicmkofupyi45vknkxz3urq6m3smdxge5ld5q"
AUTH_TYPE = "INSTANCE_PRINCIPAL" # The authentication type to use, e.g., API_KEY (default), SECURITY_TOKEN, INSTANCE_PRINCIPAL, RESOURCE_PRINCIPAL.
CONFIG_PROFILE = "DEFAULT"

# Service endpoint
endpoint = "https://inference.generativeai.ap-osaka-1.oci.oraclecloud.com"

# initialize interface
chat = ChatOCIGenAI(
  model_id="ocid1.generativeaimodel.oc1.ap-osaka-1.amaaaaaask7dceyazvmhjhprp5dtpqezqzxmkjvrhjuw3uxifk3czuany5ya",
  service_endpoint=endpoint,
  compartment_id=COMPARTMENT_ID,
  provider="cohere",
  model_kwargs=llm_params,
  auth_type=AUTH_TYPE,
  auth_profile=CONFIG_PROFILE
)

messages = [
  HumanMessage(content=args.message),
]

response = chat.invoke(messages)

# Print result
print("**************************Chat Result**************************")
print(f"Input message: {args.message}")
print(f"Used parameters: {llm_params}")
print("Response:")
print(vars(response))
