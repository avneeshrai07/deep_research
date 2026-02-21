
import boto3
import json
import os
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

def load_aws_secrets(secret_name="prod/orbit", region_name="us-east-2"):
    """
    Fetches secrets from AWS Secrets Manager and loads them into os.environ
    """
    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        print(f"Critical error in loading bossenv: {e}")
        raise e

    secret = get_secret_value_response["SecretString"]
    secrets_dict = json.loads(secret)

    # Load into os.environ
    for key, value in secrets_dict.items():
        os.environ[key] = value
        print(f"{key} = {value}")  # Debug print

if __name__ == "__main__":
    print("🔑 Fetching AWS Secrets...\n")
    load_aws_secrets()
    print("\n✅ All secrets loaded and printed successfully!")
