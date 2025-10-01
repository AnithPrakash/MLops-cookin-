from time import sleep  
from prefect_aws import S3Bucket, AwsCredentials

def create_aws_cred_blocks():
    my_aws_creds_obj = AwsCredentials(
        aws_access_key_id="1234",
        aws_secret_access_key="PLACEHOLDER",
        
    )