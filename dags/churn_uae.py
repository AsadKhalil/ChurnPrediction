from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.contrib.operators.slack_webhook_operator import SlackWebhookOperator
from airflow.operators.python import PythonOperator, PythonVirtualenvOperator

from airflow.operators.slack_operator import SlackAPIPostOperator
from datetime import datetime
from random import randint
import requests
import os
from airflow.hooks.base_hook import BaseHook
import pandas as pd
from airflow.models import Variable

args = {
    'owner': 'Asad',
    'start_date':datetime(2021, 11 ,1),
    'email': ['asad.khalil'],
    'email_on_failure':True,
}


PROJECT_LOCATION = Variable.get("PROJECT_LOCATION_UAE")
ENVIRONMENT =Variable.get("ENVIRONMENT")

slack_tokken = Variable.get("slack_tokken")
aws_access_key_id =Variable.get("aws_access_key_id")
aws_secret_access_key =Variable.get("aws_secret_access_key")
bucket_name =Variable.get("bucket_name")
start_date =Variable.get("training_start_date")
end_date =Variable.get("training_end_date")
SLACK_CONN_ID = 'slack_connection_'


text = ':red_circle: Subscription Churn Prediction UAE | \n Subscription Type : 39 AED  \n Subscription Remaining Days : 5-15'

def ds_bootstrap_func(**kwargs):
    task_instance = kwargs['task_instance']
    pipeline_run_id = kwargs['dag_run'].run_id.replace(':', '.')
    task_instance.xcom_push(key='pipeline_run_id', value=pipeline_run_id)
    date_key = 'date'
    if date_key in kwargs:
        date = kwargs['dag_run'].conf['date']   #argument date
    else:
        date =kwargs['logical_date']   #execution date

    date =date.date()

    task_instance.xcom_push(key='date', value=date)

    pass




def post_image():
    url="https://slack.com/api/files.upload"

    file_name='CHURN_PREDICTION_UAE'+'.csv'

    os.chdir(PROJECT_LOCATION)
    os.chdir('..')
    file_path =os.getcwd()+'/process/'+file_name

    with open(file_path) as fh:
        html_data = fh.read()
    data = {

        "token": slack_tokken,
        "channels": ['#subscriptions_churn_prediction'],
        "content": html_data,
        "filename": file_name,
        "initial_comment": text,
        "filetype": "csv"

    }

    response = requests.post(
         url=url, data=data,
         headers={"Content-Type": "application/x-www-form-urlencoded"})

    #response = requests.post(url=url, data=payload, params=data, files=file_upload)
    if response.status_code == 200:
        print("successfully completed post_reports_to_slack "
                      "and status code %s" % response.status_code)
    else:
        print("Failed to post report on slack channel "
                      "and status code %s" % response.status_code)



with DAG("churn_prediction_uae",
          default_args=args,
          schedule_interval='0 15 * * *',
          catchup=False,

          tags=['DataScience', 'UAE']
        ) as dag:

    bootstrap_pipeline_task = PythonOperator(
        task_id='bootstrap_pipeline_task',
        provide_context=True,
        python_callable=ds_bootstrap_func
    )

    data_download = BashOperator(
        task_id='data_download',
        bash_command = f'{ENVIRONMENT} {PROJECT_LOCATION}' + '/data_download.py --data_dir ' +f'{PROJECT_LOCATION}'

    )
    training_file = BashOperator(
        task_id='training_file',
        bash_command = f'{ENVIRONMENT} {PROJECT_LOCATION}' + '/training.py --start_date '+f'{start_date}'+' --end_date '+f'{end_date}' + ' --directory ' +f'{PROJECT_LOCATION}'

    )
    model_creation = BashOperator(
        task_id='model_creation',
        bash_command = f'{ENVIRONMENT} {PROJECT_LOCATION}' + '/model_creation.py --directory ' +f'{PROJECT_LOCATION}'

    )


    making_predictions = BashOperator(
        task_id='making_predictions',
        bash_command=f'{ENVIRONMENT} {PROJECT_LOCATION}'+"/predictionByDate.py --date {{ task_instance.xcom_pull(task_ids='bootstrap_pipeline_task',key='date') }} --directory"+f' {PROJECT_LOCATION}'

    )

    send_file_to_slack = PythonOperator(
        task_id='send_file',
        python_callable=post_image,

    )
    s3_file_upload = BashOperator(
        task_id='s3_file_upload',
        bash_command=f'{ENVIRONMENT} {PROJECT_LOCATION}'+"/s3_file_upload.py --aws_access_key "+f'{aws_access_key_id} '+"--aws_secret_access_key "+f'{aws_secret_access_key} '+"--bucket_name "+f'{bucket_name} '+"--directory "+ f'{PROJECT_LOCATION} '+ '--dag_run_id '+ "{{ task_instance.xcom_pull(task_ids='bootstrap_pipeline_task',key='pipeline_run_id') }} "
        # doc_md='Uploading workspace data including Processed, model, logs directory\'s in s3'

    )


    bootstrap_pipeline_task >> data_download >>training_file >>model_creation >>making_predictions >> [send_file_to_slack ,s3_file_upload]
