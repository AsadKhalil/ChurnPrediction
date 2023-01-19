#%%
import argparse

import dataconfig as cfg
import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path
import os
from tqdm import tqdm

#%%


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", help="Data directory location.", required=True)

args = parser.parse_args()
os.chdir(args.data_dir)
os.chdir('..')
print(os.getcwd())
store_dir = os.path.join(os.getcwd(), 'raw')
Path(store_dir).mkdir(parents=True, exist_ok=True)

#%%
MYSQL_READER_HOST = cfg.sql['host']
MYSQL_READER_USERNAME = cfg.sql['user']
MYSQL_READER_PASSWORD = cfg.sql['passwd']
MYSQL_DB_NAME = cfg.sql['db']

sqlEngine = create_engine(f'mysql+pymysql://{MYSQL_READER_USERNAME}:{MYSQL_READER_PASSWORD}@{MYSQL_READER_HOST}/{MYSQL_DB_NAME}', pool_recycle=3600)

#%%
def export_table(name, download_location):
    table = pd.read_sql(f'select /*+ MAX_EXECUTION_TIME(100000000) */ * from {name}', sqlEngine)
    table.to_csv(os.path.join(download_location, name + '.csv'), index=False)

tables = [
		'subscriptions', 'orders','user_subscriptions','meal_ratings', 'order_details','order_issue','order_histories']

for table in tqdm(tables):
    print(f'\t => \t Storing {table}')
    export_table(table, store_dir)
# %%
