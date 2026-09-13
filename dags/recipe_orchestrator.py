from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data_engineer',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
        'pyspark_recipe_pipeline',
        default_args=default_args,
        description='Orchestrates the Parquet conversion job',
        schedule_interval='@daily',
        start_date=datetime(2026, 9, 12),
        catchup=False,
        tags=['data_engineering','pyspark']
    ) as dag:
        verify_raw_data = BashOperator(
        task_id='verify_raw_data',
        bash_command='test -f /opt/airflow/data/raw/RecipeNLG_dataset.csv && echo "Data exists."'
        )

        execute_pipeline = BashOperator(
            task_id='execute_pyspark',
            bash_command='python3 /opt/airflow/dags/pipeline.py',
            cwd='/opt/airflow'
        )
        verify_raw_data >> execute_pipeline