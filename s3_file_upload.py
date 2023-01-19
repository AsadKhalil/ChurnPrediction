import boto3
from botocore.exceptions import NoCredentialsError
from botocore.exceptions import ClientError
import sys
import os
import argparse
import logging
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("--aws_access_key", help="AWS ACCESS Key", required=True)
parser.add_argument("--aws_secret_access_key", help="Aws SECRET_KEY",required=True)
parser.add_argument("--bucket_name", help="S3 bucket name", required=True)
parser.add_argument("--directory", help="Project Location", required=True)
parser.add_argument("--dag_run_id", help="Dag Run ID For Folder Creation on S3", required=True)


args = parser.parse_args()

ACCESS_KEY = args.aws_access_key
SECRET_KEY = args.aws_secret_access_key
bucket_name = args.bucket_name
directory = args.directory
folder_name = args.dag_run_id

log_dir_path = os.path.join(directory, '..', 'logs')
Path(log_dir_path).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    handlers = [ logging.FileHandler(Path(log_dir_path).joinpath('file_upload'+'.log'),mode='w+') ]
)

logger = logging.getLogger(__name__)
logger.info('directory : %s',directory)
logger.info('folder name %s',folder_name)

print(directory)
print(folder_name)

client_s3 = boto3.client(
    's3',
    aws_access_key_id=  ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY

)

u_folders = ['process','model','raw','logs']



for i in u_folders:
    os.chdir(directory)
    os.chdir('..')
    file_path =os.getcwd()+'/'+i+'/'
    print(file_path)
    for file in os.listdir(file_path):
        if not file.startswith('~'):
            try:
                print("Uploading File" +file)
                logger.info('Uploading File: %s', file)
                client_s3.upload_file(
                os.path.join(file_path,file),
                bucket_name,"UAE"+'/'+folder_name+'/'+i+'/'+file
                )
            except ClientError as e:
                print('CredentialsError'+e)
                logger.info('CredentialsError: %s', e)
