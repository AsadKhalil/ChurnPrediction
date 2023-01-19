import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import os
import argparse
import logging
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--start_date", help="Start training Date Period YYYY-MM-DD", required=True)
parser.add_argument("--end_date", help="End training Date Period YYYY-MM-DD", required=True)
parser.add_argument("--directory", help="PROJECT LOCATION",required=True)

args = parser.parse_args()

#Getting command line arguments
start_date =args.start_date
end_date =args.end_date
directory = args.directory

log_dir_path = os.path.join(directory, '..', 'logs')
Path(log_dir_path).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    handlers = [ logging.FileHandler(Path(log_dir_path).joinpath('training'+'.log'),mode='w+') ]
)

logger = logging.getLogger(__name__)

logger.info("Directory:   %s" ,directory)

os.chdir(directory)
os.chdir('..')
print(os.getcwd())
#Setting Directories

data = os.getcwd()+'/raw'

training =os.getcwd()
Path(training+"/training").mkdir(parents=True, exist_ok=True)
training =training+"/training/"


print("File Reading")
logger.info('File Reading: %s ')

orders = pd.read_csv(data+'/orders.csv', usecols=['id', 'schedule_on', 'status', 'user_subscription_id'])
user_subscriptions = pd.read_csv(data+'/user_subscriptions.csv', usecols=['id', 'user_id', 'start_date', 'end_date', 'cancelled_at', 'is_trial', 'subscription_id','promo_id'])
order_issues = pd.read_csv(data+'/order_issue.csv', usecols=['order_id', 'issue_id'])
meal_ratings = pd.read_csv(data+'/meal_ratings.csv', usecols=['order_id', 'rating'])
subscriptions = pd.read_csv(data+'/subscriptions.csv', usecols=['id', 'fee'])
order_details = pd.read_csv(data+'/order_details.csv', usecols=['order_id', 'user_delivery_time', 'total_saving'])
order_histories = pd.read_csv(data+'/order_histories.csv', usecols=['order_id', 'status', 'expected_time', 'created_at'])

#DATA CLEANING
print(user_subscriptions.head())
user_subscriptions= user_subscriptions.loc[ (user_subscriptions['promo_id'].isna())]

order_histories = order_histories.sort_values(by=['order_id','status'], ascending=False)
order_histories.rename({'status':'history_status'},axis=1,inplace=True)
order_histories['expected_time']=pd.to_datetime(order_histories['expected_time'])
order_histories['created_at']=pd.to_datetime(order_histories['created_at'])

order_histories = order_histories.drop_duplicates(subset='order_id', keep="first")
order_histories.drop_duplicates(inplace=True)
orders['schedule_on'] = pd.to_datetime(orders['schedule_on'])
user_subscriptions['end_date'] = pd.to_datetime(user_subscriptions['end_date'])
user_subscriptions['start_date'] = pd.to_datetime(user_subscriptions['start_date'])
user_subscriptions['cancelled_at'] =pd.to_datetime(user_subscriptions['cancelled_at'])

## Selecting NON TRIAL SUBSCRIPTIONS ONLY
user_subscriptions = user_subscriptions.loc[user_subscriptions['is_trial'] == 0]

# User subscription with Subscription FEE
user_subscriptions =user_subscriptions.merge(subscriptions,how='inner',left_on=['subscription_id'],right_on=['id'])

user_subscriptions.drop(['id_y'],axis=1,inplace=True)
user_subscriptions.rename({'id_x':'id'}, axis=1, inplace=True)

# # Mostly subscriptions are of 39AED so using them  SUBSCRIPTION TYPE
user_subscriptions =user_subscriptions.loc[(user_subscriptions['subscription_id'] == 2) | (user_subscriptions['subscription_id'] == 6)]

user_subscriptions.sort_values(['user_id','start_date'],inplace=True)
user_subscriptions['next_sub'] = user_subscriptions.groupby('user_id')['start_date'].shift(-1)

user_subscriptions['subscription_type'] = 'renew'

user_subscriptions.loc[user_subscriptions.groupby('user_id')['subscription_type'].head(1).index, 'subscription_type'] = 'new'
user_subscriptions.loc[((user_subscriptions['next_sub']-user_subscriptions['start_date']).astype('timedelta64[M]') >2),'subscription_type'] = 'new'
user_subscriptions.loc[user_subscriptions['cancelled_at'].isnull(),'actual_Y'] = 'renew'

last_Sub = user_subscriptions.groupby(['user_id'])['id','user_id','start_date','end_date','cancelled_at','is_trial'].tail(1)
last_Sub =last_Sub.loc[last_Sub['cancelled_at'].isna()]

today = datetime.today().strftime('%Y-%m-%d')

last_Sub.loc[( pd.to_datetime(today)- last_Sub['end_date']).dt.days > 10,'no_renew']='other'

last_Sub.drop(['user_id','start_date','end_date','cancelled_at','is_trial'],axis=1,inplace=True)


user_subscriptions= user_subscriptions.merge(last_Sub,how='left',on='id')
user_subscriptions.sort_values(['user_id','end_date'],ascending=[True,True],inplace=True)

# calculate shift of cancelled_At
user_subscriptions['shift'] = user_subscriptions.groupby('user_id')['cancelled_at'].shift(-1)

user_subscriptions['canceldays'] =abs(user_subscriptions['start_date'] - user_subscriptions['cancelled_at']).dt.days

user_subscriptions['canceldays'] =abs(user_subscriptions['start_date'] - user_subscriptions['cancelled_at']).dt.days

user_subscriptions.loc[user_subscriptions['no_renew']=='other','actual_Y']='other'
user_subscriptions.loc[user_subscriptions['cancelled_at'].notnull(), 'actual_Y'] = 'cancel'
user_subscriptions.loc[(user_subscriptions['shift'] - user_subscriptions['end_date']).dt.days < 10,'actual_Y'] = 'cancel'

user_subscriptions.drop(user_subscriptions[user_subscriptions['actual_Y'] =='other'].index, inplace = True)



user_subscriptions = user_subscriptions[user_subscriptions['start_date'].between(start_date, end_date)]

logger.info('ACTUAL Y: %s ',user_subscriptions['actual_Y'].value_counts())

logger.info('Removing Subscription Remaining Days > 31: %s ')

rem = (pd.to_datetime(user_subscriptions['end_date'])) - (pd.to_datetime(user_subscriptions['start_date']))
user_subscriptions.drop(user_subscriptions[rem.dt.days >31].index, inplace=True)

#Joining with Orders
logger.info('Joining with Orders: %s ')

user_subscriptions = user_subscriptions.merge(orders,how='left',left_on=['id'],right_on=['user_subscription_id'])


user_subscriptions.sort_values(['id_x','schedule_on'],inplace=True)

user_subscriptions = user_subscriptions[(user_subscriptions.schedule_on >= user_subscriptions.start_date) &
                                        (user_subscriptions.schedule_on <= user_subscriptions.end_date)]
user_subscriptions.drop(['is_trial','next_sub','shift','canceldays'],axis=1,inplace=True)


user_subscriptions.drop(['user_subscription_id','subscription_id'],axis=1,inplace=True)
user_subscriptions.rename({'id_x':'user_subscription_id','id_y':'order_id'}, axis=1, inplace=True)

#LAST_ORDER_DAYS Feature
logger.info('LAST_ORDER_DAYS Feature: %s ')

order_issues.drop_duplicates(inplace=True)
meal_ratings.drop_duplicates(inplace=True)

last = user_subscriptions.groupby('user_subscription_id')['user_subscription_id','schedule_on','order_id'].tail(1)

last = last.merge(order_issues,how='left',left_on=['order_id'],right_on=['order_id'])
#Selecting order with
last = last.merge(meal_ratings,how='left',left_on=['order_id'],right_on=['order_id'])
# #Selecting order with
last = last.merge(order_histories,how='left',left_on=['order_id'],right_on=['order_id'])

last['diff'] =last['expected_time'].sub(last['created_at']).dt.total_seconds().div(60)

last['last_failed_order']=False
last.loc[ last['history_status']!= 5,'last_failed_order'] = True

last['last_order_OTIF'] =True
last.loc[( (last['diff']>15) | (last['diff']< -15) ),'last_order_OTIF'] = False

last.loc[ last['expected_time'].isna(),'last_order_OTIF'] = False

#fill -1 rating with nan value
last.loc[last['rating'] == -1, 'rating'] = np.nan
num = last['rating'].mean()
last['rating'].fillna(num, inplace=True)

last['last_order_issue'] = False
last.loc[ last['issue_id'].notnull(),'last_order_issue'] = True

last.drop(['order_id','issue_id','history_status','expected_time','created_at','diff'],axis=1,inplace=True)
last.rename({'rating':'last_order_rating'}, axis=1, inplace=True)


user_subscriptions = user_subscriptions.merge(last,how='inner',left_on=['user_subscription_id'],right_on=['user_subscription_id'])
#selecting orders with subscriptionss
user_subscriptions.rename({'schedule_on_x':'schedule_on','schedule_on_y':'last_order_sub'}, axis=1, inplace=True)

user_subscriptions.drop_duplicates(inplace=True)

user_subscriptions.drop_duplicates(['user_subscription_id','user_id','order_id'],inplace=True)

user_subscriptions['no_order_days'] =  pd.to_datetime( user_subscriptions['end_date'])  - pd.to_datetime( user_subscriptions['last_order_sub'])
logger.info('Subscription Remaining Days Feature: %s ')

user_subscriptions['remaining_days_sub'] =  pd.to_datetime( user_subscriptions['end_date'])  - pd.to_datetime( user_subscriptions['last_order_sub'])

order_details.drop_duplicates(inplace=True)

logger.info('Total Saving in Subscription Feature: %s ')

user_subscriptions = user_subscriptions.merge(order_details,how='left',left_on=['order_id'],right_on=['order_id'])
#selecting orders with subscriptionss
usub=user_subscriptions.loc[user_subscriptions['status'] ==5]

usub['ac']=usub.groupby(['user_subscription_id'])['total_saving'].transform('sum')
usub['saving'] =(usub['ac'] - usub['fee'])
saving = usub.groupby('user_subscription_id')['user_subscription_id','saving'].tail(1)


def dd(x):
    return x[x== 8].count()

logger.info('Total Orders Canceled Feature: %s ')

canceled_orders = user_subscriptions.groupby(['user_subscription_id','schedule_on'])['status'].apply(dd).reset_index(name="total_cancel")
canceled_orders['order_cancel'] = canceled_orders['total_cancel'] > 0
logger.info('Total Order Delivered Feature: %s ')

delivered_orders = user_subscriptions.groupby(['user_subscription_id'])['status'].apply(lambda  x:x[x == 5].count()  + x[x == 10].count()).reset_index(name='delivered_orders')

def payment(x):
    return x[x== 6].count() + x[x== 7].count()

logger.info('Payment Declined Orders Feature: %s ')

payment_declined_orders = user_subscriptions.groupby(['user_subscription_id','schedule_on'])['status'].apply(payment).reset_index(name='payment_declined_orders')
payment_declined_orders['payment_fail'] = payment_declined_orders['payment_declined_orders'] > 0

logger.info('Total Order Issues Feature: %s ')

order_issues = order_issues.drop_duplicates(subset='order_id', keep="first")
user_subscriptions = user_subscriptions.merge(order_issues,how='left',left_on=['order_id'],right_on=['order_id'])
#Selecting order with

def orderIssue(x):
    return x.count()

total_order_issue = user_subscriptions.groupby(['user_subscription_id','schedule_on'])['issue_id'].apply(orderIssue).reset_index(name='total_order_issues')
total_order_issue['issue_count'] = total_order_issue['total_order_issues'] > 0

user_subscriptions = user_subscriptions.merge(meal_ratings,how='left',left_on=['order_id'],right_on=['order_id'])
#Selecting order with
logger.info('Average Rating Feature: %s ')

#fill -1 rating with nan value
user_subscriptions.loc[user_subscriptions['rating'] == -1, 'rating'] = np.nan
#fill nan value with mean rating in each subcription id group
user_subscriptions['rating']=user_subscriptions.groupby('user_subscription_id')['rating'].apply(lambda x:x.fillna(x.mean()))

total_avg_rating = user_subscriptions.groupby('user_subscription_id')['rating'].apply(lambda  x: x.mean()).reset_index(name='avg_rating')
total_avg_rating=total_avg_rating.apply(lambda x: x.fillna(x.mean()))
logger.info('OTIF FEATURE: %s ')
order_histories =order_histories[order_histories['history_status'].eq(5)]

user_subscriptions = user_subscriptions.merge(order_histories,how='left',left_on=['order_id'],right_on=['order_id'])
user_subscriptions['diff'] =user_subscriptions['expected_time'].sub(user_subscriptions['created_at']).dt.total_seconds().div(60)
user_subscriptions['otif'] =True
user_subscriptions.loc[( (user_subscriptions['diff']>15) | (user_subscriptions['diff']< -15) ),'otif'] = False

def otif_Late(x):
    flse = x[x ==False].count()
    total = x.count()
    return flse/total

otif = user_subscriptions.groupby(['user_subscription_id'])['otif'].apply(otif_Late).reset_index(name="otif_late")

otif.sort_values('otif_late')
user_subscriptions.drop_duplicates(['user_subscription_id','user_id','order_id'],inplace=True)

data = total_order_issue.merge(payment_declined_orders,on=['user_subscription_id','schedule_on']).merge(
        canceled_orders,on=['user_subscription_id','schedule_on'])

data =data.groupby(['user_subscription_id']).sum().reset_index()

data = data.merge(user_subscriptions,on='user_subscription_id').merge(saving,on='user_subscription_id').merge(
        delivered_orders,on='user_subscription_id').merge(total_avg_rating,on='user_subscription_id').merge(
        otif,on='user_subscription_id')

data['no_order_days'] =data['no_order_days'].dt.days
data['remaining_days_sub']=data['remaining_days_sub'].dt.days

data.to_csv(training+"/training.csv",encoding='utf-8',index=False)
