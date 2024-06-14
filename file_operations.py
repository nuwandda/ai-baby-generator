import os
import boto3
from botocore.exceptions import NoCredentialsError


def upload_folder_to_s3(folder_path, bucket_name, region_name='eu-west-2'):
    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    if not aws_access_key or not aws_secret_key:
        print('AWS credentials not set in environment variables')
        return
    
    # Initialize the S3 client
    s3 = boto3.client('s3', 
                      aws_access_key_id=aws_access_key,
                      aws_secret_access_key=aws_secret_key,
                      region_name=region_name)
    
    # Walk through the directory
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            # Full file path
            file_path = os.path.join(root, filename)
            # Relative path for the key in S3
            relative_path = 'aiBaby/reference_photos/' + os.path.relpath(file_path, folder_path)
            try:
                # Upload the file to S3
                s3.upload_file(file_path, bucket_name, relative_path)
                print(f'Successfully uploaded {relative_path} to {bucket_name}')
            except FileNotFoundError:
                print(f'The file {file_path} was not found')
            except NoCredentialsError:
                print('Credentials not available')
            except Exception as e:
                print(f'An error occurred: {e}')


def download_folder_from_s3(bucket_name, folder_name, local_dir, region_name='eu-west-2'):
    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    if not aws_access_key or not aws_secret_key:
        print('AWS credentials not set in environment variables')
        return
    
    # Initialize the S3 client
    s3 = boto3.client('s3', 
                      aws_access_key_id=aws_access_key,
                      aws_secret_access_key=aws_secret_key,
                      region_name=region_name)
    
    try:
        # List objects within the specified folder
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=folder_name)
        
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    # Get the object key
                    key = obj['Key']
                    # Define the local path
                    local_path = os.path.join(local_dir, os.path.relpath(key, folder_name))
                    
                    # Create local directories if they do not exist
                    if not os.path.exists(os.path.dirname(local_path)):
                        os.makedirs(os.path.dirname(local_path))
                    
                    # Download the file
                    s3.download_file(bucket_name, key, local_path)
                    print(f'Successfully downloaded {key} to {local_path}')
    except NoCredentialsError:
        print('Credentials not available')
    except Exception as e:
        print(f'An error occurred: {e}')


if __name__ == '__main__':
    # Folder containing images
    folder_path = 'reference_photos'
    
    # S3 bucket name
    bucket_name = 'aibabygenerator-models'
    
    # Upload the entire folder to S3
    upload_folder_to_s3(folder_path, bucket_name)
