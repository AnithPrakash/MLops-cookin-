from time import sleep  
from prefect_aws import S3Bucket, AwsCredentials

def create_aws_cred_blocks():
    my_aws_creds_obj = AwsCredentials(
        aws_access_key_id="",
        aws_secret_access_key="",
        region_name="us-east-2"
    )
    my_aws_creds_obj.save(name="my-aws-creds", overwrite=True)


def create_s3_bucket_block():
    aws_creds = AwsCredentials.load("my-aws-creds")
    my_s3_buckets_obj=S3Bucket(
        bucket_name="myawsbucket20345", credentials=aws_creds
    )
    my_s3_buckets_obj.save(name="s3-bucket-example", overwrite=True)

if __name__=="__main__":
    create_aws_cred_blocks()
    sleep(5)
    create_s3_bucket_block()
