import logging
import argparse
import os
import numpy  as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import *

from sklearn import preprocessing
import pickle
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")


# Pass the directory where the data is located
def fileReading(directory):
    print("File Reading")
    logger.info('File Reading: %s ')

    orders = pd.read_csv(directory+'/orders.csv', usecols=['id', 'schedule_on', 'status', 'user_subscription_id'])
    user_subscriptions = pd.read_csv(directory+'/user_subscriptions.csv', usecols=['id', 'user_id', 'start_date', 'end_date', 'cancelled_at', 'is_trial', 'subscription_id'])
    order_issues = pd.read_csv(directory+'/order_issue.csv', usecols=['order_id', 'issue_id'])
    meal_ratings = pd.read_csv(directory+'/meal_ratings.csv', usecols=['order_id', 'rating'])
    subscriptions = pd.read_csv(directory+'/subscriptions.csv', usecols=['id', 'fee'])
    order_details = pd.read_csv(directory+'/order_details.csv', usecols=['order_id', 'user_delivery_time', 'total_saving'])
    order_histories = pd.read_csv(directory+'/order_histories.csv', usecols=['order_id', 'status', 'expected_time', 'created_at'])

    return orders,user_subscriptions,order_issues,meal_ratings,subscriptions,order_details,order_histories

#Passing the DataFrames for Cleaning
def dataCleaning(order_histories,orders,user_subscriptions):
    order_histories = order_histories.sort_values(by=['order_id', 'status'], ascending=False)

    order_histories.rename({'status': 'history_status'}, axis=1, inplace=True)

    order_histories['expected_time'] = pd.to_datetime(order_histories['expected_time'])
    order_histories['created_at'] = pd.to_datetime(order_histories['created_at'])

    order_histories = order_histories.drop_duplicates(subset='order_id', keep="first")
    order_histories.drop_duplicates(inplace=True)

    orders['schedule_on'] = pd.to_datetime(orders['schedule_on'])

    user_subscriptions['end_date'] = pd.to_datetime( user_subscriptions['end_date'])
    user_subscriptions['start_date'] = pd.to_datetime(user_subscriptions['start_date'])
    user_subscriptions['cancelled_at'] = pd.to_datetime(user_subscriptions['cancelled_at'])

    # NEW & RENEWED FEATURE
    user_subscriptions = user_subscriptions.loc[user_subscriptions['is_trial'] == 0]
    user_subscriptions.sort_values(['user_id', 'start_date'], inplace=True)

    user_subscriptions['next_sub'] = user_subscriptions.groupby('user_id')['start_date'].shift(-1)

    user_subscriptions['subscription_type'] = 'renew'

    user_subscriptions.loc[user_subscriptions.groupby('user_id')['subscription_type'].head(1).index, 'subscription_type'] = 'new'
    user_subscriptions.loc[((user_subscriptions['next_sub']-user_subscriptions['start_date']).astype('timedelta64[M]') > 2), 'subscription_type'] = 'new'

    return order_histories,orders,user_subscriptions

#user subscriptions Data
def userSubscriptionsDataFiltering(user_subscriptions,subscriptions,active_subscription_date):
    # User Subscriptions till filter
    user_subscriptions = user_subscriptions[user_subscriptions['end_date'] >= active_subscription_date]
    # User subscription with Subscription FEE

    user_subscriptions = user_subscriptions.merge(subscriptions, how='inner', left_on=['subscription_id'], right_on=['id'])
    user_subscriptions.drop(['id_y'], axis=1, inplace=True)
    user_subscriptions.rename({'id_x': 'id'}, axis=1, inplace=True)
    user_subscriptions = user_subscriptions.loc[(user_subscriptions['subscription_id'] == 2)]

    user_subscriptions.loc[user_subscriptions['cancelled_at'].isnull(), 'actual_Y'] = 'renew'
    last_Sub = user_subscriptions.groupby(['user_id'])['id', 'user_id', 'start_date', 'end_date', 'cancelled_at', 'is_trial'].tail(1)
    last_Sub = last_Sub.loc[last_Sub['cancelled_at'].isna()]

    today = datetime.today().strftime('%Y-%m-%d')
    last_Sub.loc[(pd.to_datetime(today) - last_Sub['end_date']).dt.days > 10, 'no_renew'] = 'other'
    last_Sub.drop(['user_id', 'start_date', 'end_date','cancelled_at', 'is_trial'], axis=1, inplace=True)

    user_subscriptions = user_subscriptions.merge(last_Sub, how='left', on='id')
    user_subscriptions.sort_values(['user_id', 'end_date'], ascending=[True, True], inplace=True)
    user_subscriptions['shift'] = user_subscriptions.groupby('user_id')['cancelled_at'].shift(-1)
    user_subscriptions['canceldays'] = abs(user_subscriptions['start_date'] - user_subscriptions['cancelled_at']).dt.days
    # removing Subscriptions which are cancelled within 3 days
    user_subscriptions.drop(user_subscriptions[user_subscriptions['canceldays'] <=3].index,inplace=True)

    user_subscriptions.loc[user_subscriptions['no_renew']== 'other', 'actual_Y'] = 'other'

    user_subscriptions.loc[user_subscriptions['cancelled_at'].notnull(), 'actual_Y'] = 'cancel'
    user_subscriptions.loc[(user_subscriptions['shift'] -user_subscriptions['end_date']).dt.days < 5, 'actual_Y'] = 'cancel'

    user_subscriptions.drop(user_subscriptions[user_subscriptions['actual_Y'] == 'other'].index, inplace=True)
    # only CANCELLED

    # Removing Subscription period greater than 31 (NOT USING PAUSED SUBSCRIPTIONS)
    rem = (pd.to_datetime(user_subscriptions['end_date'])) - (pd.to_datetime(user_subscriptions['start_date']))
    user_subscriptions.drop(user_subscriptions[rem.dt.days > 31].index, inplace=True)

    return user_subscriptions

#calculating Last Order Day Features
def lastOrderDayFeatures(order_issues,meal_ratings,user_subscriptions,order_histories,Date):

    order_issues.drop_duplicates(inplace=True)
    meal_ratings.drop_duplicates(inplace=True)
    ##
    last_N_order_days = user_subscriptions.sort_values(['user_subscription_id', 'schedule_on']).groupby('user_subscription_id')['user_subscription_id', 'schedule_on', 'order_id'].tail(1)

    last_N_order_days = last_N_order_days.merge(order_issues, how='left', left_on=['order_id'], right_on=['order_id'])
    last_N_order_days = last_N_order_days.merge(meal_ratings, how='left', left_on=['order_id'], right_on=['order_id'])
    last_N_order_days = last_N_order_days.merge(order_histories, how='left', left_on=['order_id'], right_on=['order_id'])

    last_N_order_days['diff'] = last_N_order_days['expected_time'].sub(last_N_order_days['created_at']).dt.total_seconds().div(60)
    last_N_order_days['last_failed_order'] = False
    last_N_order_days.loc[last_N_order_days['history_status']!= 5, 'last_failed_order'] = True

    last_N_order_days['last_order_OTIF'] = True
    last_N_order_days.loc[((last_N_order_days['diff'] > 15) | (last_N_order_days['diff'] < -15)), 'last_order_OTIF'] = False

    last_N_order_days.loc[last_N_order_days['expected_time'].isna(), 'last_order_OTIF'] = False

    # fill -1 rating with nan value
    last_N_order_days.loc[last_N_order_days['rating'] == -1, 'rating'] = np.nan
    num = last_N_order_days['rating'].mean()
    last_N_order_days['rating'].fillna(num, inplace=True)

    last_N_order_days['last_order_issue'] = False
    last_N_order_days.loc[last_N_order_days['issue_id'].notnull(), 'last_order_issue'] = True

    last_N_order_days['no_order_days'] = Date -  pd.to_datetime(last_N_order_days['schedule_on'])
    last_N_order_days.drop(['order_id', 'issue_id', 'history_status',
                           'expected_time', 'created_at', 'diff'], axis=1, inplace=True)
    last_N_order_days.rename(
        {'rating': 'last_order_rating'}, axis=1, inplace=True)

    print("LAST_ORDER_DAYS Feature")
    logger.info('LAST_ORDER_DAYS Feature: %s ')

    return last_N_order_days

#calaculating subscription remaining day feature
def subRemainingDays(user_subscriptions,Date):
    print('Subscription Remaining Days Feature')
    logger.info('Subscription Remaining Days Feature: %s ')

    sub_remaining_days = user_subscriptions.sort_values('schedule_on').groupby('user_subscription_id').tail(1)
    sub_remaining_days['remaining_days_sub'] = pd.to_datetime(sub_remaining_days['end_date']) - pd.to_datetime(Date)
    sub_remaining_days = sub_remaining_days[['user_subscription_id', 'remaining_days_sub']].copy()

    return sub_remaining_days

#calculating Saving Features
def savingFeature(order_details,user_subscriptions):
    print("Total Saving in Subscription Feature")
    logger.info('Total Saving in Subscription Feature: %s ')

    order_details.drop_duplicates(inplace=True)
    user_subscriptions = user_subscriptions.merge(order_details, how='inner', left_on=['order_id'], right_on=['order_id'])
    # selecting orders with subscriptions
    usub = user_subscriptions.loc[user_subscriptions['status'] == 5]
    usub['ac'] = usub.groupby(['user_subscription_id'])['total_saving'].transform('sum')
    usub['saving'] = (usub['ac'] - usub['fee'])
    saving = usub.groupby('user_subscription_id')['user_subscription_id', 'saving'].tail(1)

    return saving

def cancel_orders(x):
    return x[x == 8].count()

#calculating total cancel order and delivered orders
def cancelOrderTotalOrders(user_subscriptions):
    print("Total Orders Canceled Feature")
    logger.info('Total Orders Canceled Feature: %s ')



    canceled_orders = user_subscriptions.groupby(['user_subscription_id', 'schedule_on'])['status'].apply(cancel_orders).reset_index(name="total_cancel")
    canceled_orders['order_cancel'] = canceled_orders['total_cancel'] > 0

    print("Total Order Delivered Feature")
    logger.info('Total Order Delivered Feature: %s ')

    delivered_orders = user_subscriptions.groupby(['user_subscription_id'])['status'].apply(lambda x: x[x == 5].count() + x[x == 10].count()).reset_index(name='delivered_orders')

    return canceled_orders,delivered_orders

def payment(x):
    return x[x == 6].count() + x[x == 7].count()

def orderIssue(x):
    return x.count()

#Calculating total payment Declined Orders
def paymentDeclinedOrderIssues(user_subscriptions,order_issues):

    print("Payment Declined Orders Feature")
    logger.info('Payment Declined Orders Feature: %s ')


    payment_declined_orders = user_subscriptions.groupby(['user_subscription_id', 'schedule_on'])['status'].apply(payment).reset_index(name='payment_declined_orders')
    payment_declined_orders['payment_fail'] = payment_declined_orders['payment_declined_orders'] > 0

    print("Total Order Issues Feature")
    logger.info('Total Order Issues Feature: %s ')

    order_issues = order_issues.drop_duplicates(subset='order_id', keep="first")
    user_subscriptions = user_subscriptions.merge(order_issues, how='left', left_on=['order_id'], right_on=['order_id'])

    total_order_issue = user_subscriptions.groupby(['user_subscription_id', 'schedule_on'])['issue_id'].apply(orderIssue).reset_index(name='total_order_issues')
    total_order_issue['issue_count'] = total_order_issue['total_order_issues'] > 0

    return payment_declined_orders,total_order_issue

#calculating Average rating in Subscriptions
def averageRatingInSubscription(user_subscriptions,meal_ratings):
    print("Average Rating Feature")
    logger.info('Average Rating Feature: %s ')

    user_subscriptions = user_subscriptions.merge(meal_ratings, how='left', left_on=['order_id'], right_on=['order_id'])
    user_subscriptions.loc[user_subscriptions['rating']== -1, 'rating'] = np.nan
    # fill nan value with mean rating in each subcription id group
    user_subscriptions['rating'] = user_subscriptions.groupby('user_subscription_id')['rating'].apply(lambda x: x.fillna(x.mean()))

    total_avg_rating = user_subscriptions.groupby('user_subscription_id')['rating'].apply(lambda x: x.mean()).reset_index(name='avg_rating')

    total_avg_rating = total_avg_rating.apply(lambda x: x.fillna(x.mean()))

    return total_avg_rating
#calculating Otif Rate
def otifRate(order_histories,user_subscriptions):
    print("OTIF FEATURE")
    logger.info('OTIF FEATURE: %s ')

    order_histories = order_histories[order_histories['history_status'].eq(5)]

    user_subscriptions = user_subscriptions.merge(order_histories, how='left', left_on=['order_id'], right_on=['order_id'])

    user_subscriptions['diff'] = user_subscriptions['expected_time'].sub(user_subscriptions['created_at']).dt.total_seconds().div(60)
    user_subscriptions['otif'] = True
    user_subscriptions.loc[((user_subscriptions['diff'] > 15) | (user_subscriptions['diff'] < -15)), 'otif'] = False

    def otif_Late(x):
        flse = x[x == False].count()
        total = x.count()
        return flse/total

    otif = user_subscriptions.groupby(['user_subscription_id'])['otif'].apply(otif_Late).reset_index(name="otif_late")
    return otif


def prediction(Date,directory):

    Date = pd.to_datetime(Date)

    logger.info('Date: %s ', Date)

    os.chdir(directory)
    os.chdir('..')
    print(os.getcwd())
    #Setting Directories

    data = os.getcwd()+'/raw'
    model = os.getcwd()+'/model'
    processed =os.getcwd()
    Path(processed+"/process").mkdir(parents=True, exist_ok=True)
    processed =processed+"/process/"


    orders,user_subscriptions,order_issues,meal_ratings,subscriptions,order_details,order_histories = fileReading(data)

    order_histories,orders,user_subscriptions =dataCleaning(order_histories,orders,user_subscriptions)

    user_subscriptions=userSubscriptionsDataFiltering(user_subscriptions,subscriptions,Date)


    user_subscriptions = user_subscriptions.merge(orders, how='left', left_on=['id'], right_on=['user_subscription_id'])
    # selecting orders with subscriptions
    user_subscriptions = user_subscriptions[(user_subscriptions.schedule_on >= user_subscriptions.start_date) & (user_subscriptions.schedule_on <= Date)]
    user_subscriptions.drop(['is_trial', 'next_sub', 'shift', 'canceldays', ], axis=1, inplace=True)
    user_subscriptions.drop(['user_subscription_id', 'subscription_id'], axis=1, inplace=True)
    user_subscriptions.rename({'id_x': 'user_subscription_id', 'id_y': 'order_id'}, axis=1, inplace=True)

    #LAST DAYS FEATURES
    last_N_order_days =lastOrderDayFeatures(order_issues,meal_ratings,user_subscriptions,order_histories,Date)
    #Subscription Remaining Days Features
    sub_remaining_days= subRemainingDays(user_subscriptions,Date)
    user_subscriptions.drop_duplicates(['user_subscription_id', 'user_id', 'order_id'], inplace=True)

    #calaculating Saving Feature
    saving= savingFeature(order_details,user_subscriptions)

    canceled_orders,delivered_orders =  cancelOrderTotalOrders(user_subscriptions)

    payment_declined_orders,total_order_issue =paymentDeclinedOrderIssues(user_subscriptions,order_issues)

    total_avg_rating=averageRatingInSubscription(user_subscriptions,meal_ratings)

    otif = otifRate(order_histories,user_subscriptions)


    print("JOINING ALL FEATURES")
    logger.info('JOINING ALL FEATURES: %s ')

    data = total_order_issue.merge(payment_declined_orders, on=['user_subscription_id', 'schedule_on']).merge(canceled_orders, on=['user_subscription_id', 'schedule_on'])
    data = data.groupby(['user_subscription_id']).sum().reset_index()

    data = data.merge(user_subscriptions, on='user_subscription_id').merge(saving, on='user_subscription_id').merge(
        delivered_orders, on='user_subscription_id').merge(total_avg_rating, on='user_subscription_id').merge(
        otif, on='user_subscription_id').merge(last_N_order_days, on='user_subscription_id').merge(
        sub_remaining_days, on='user_subscription_id')

    data['no_order_days'] = data['no_order_days'].dt.days
    data['remaining_days_sub'] = data['remaining_days_sub'].dt.days

    data = data[['user_subscription_id', 'last_order_rating', 'last_failed_order', 'last_order_OTIF', 'last_order_issue',
                'subscription_type', 'saving', 'delivered_orders', 'avg_rating', 'total_order_issues', 'payment_declined_orders',
                 'total_cancel', 'no_order_days', 'remaining_days_sub', 'otif_late',
                 'order_cancel', 'payment_fail', 'issue_count','cancelled_at', 'actual_Y']].copy()

    data['last_order_rating'] = data['last_order_rating'].round(1)
    data['avg_rating'] = data['avg_rating'].round(1)
    data['saving'] = data['saving'].round(1)
    data['otif_late'] = data['otif_late'] * 100
    data['otif_late'] = data['otif_late'].round(1)
    data.drop_duplicates(inplace=True)
    # load model from file

    loaded_model = pickle.load(open(model +"/model.dat", "rb"))

    print(data.head())
    lbl = preprocessing.LabelEncoder()
    data['subscription_type'] = lbl.fit_transform(data['subscription_type'].astype(str))


    X = data[['last_failed_order', 'last_order_OTIF', 'last_order_issue', 'subscription_type', 'delivered_orders',
              'avg_rating', 'total_order_issues', 'payment_declined_orders', 'total_cancel', 'no_order_days', 'otif_late']]

    Y = data[['actual_Y']]

    # make predictions for test data
    y_pred = loaded_model.predict(X)
    # predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(Y, y_pred)

    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("Precision: %.2f%%" % (precision_score(
        Y, y_pred, average='binary', pos_label='cancel') * 100.0))
    print("Recall: %.2f%%" % (recall_score(
        Y, y_pred, average='binary', pos_label='cancel') * 100.0))
    print("F1-score: %.2f%%" %
          (f1_score(Y, y_pred, average='binary', pos_label='cancel') * 100.0))

    logger.info("Accuracy: %s" , (accuracy * 100.0))
    logger.info("Precision: %s" ,
    (precision_score(Y, y_pred, average='binary', pos_label='cancel') * 100.0))

    logger.info("Recall: %s",
    (recall_score(Y, y_pred, average='binary', pos_label='cancel') * 100.0))

    logger.info("F1-score:  %s" ,
          (f1_score(Y, y_pred, average='binary', pos_label='cancel') * 100.0))

    prob = loaded_model.predict_proba(X)
    dataset = pd.DataFrame( {'cancel_prob': prob[:, 0], 'renew_prob': prob[:, 1]})
    data = data.reset_index(drop=True)
    dataset = dataset.reset_index(drop=True)
    data = pd.concat([data, dataset['cancel_prob'],
                     dataset['renew_prob']], axis=1)
    data['predicted_Y'] = y_pred

    name = str(Date.year) + "-"+ str(Date.month)+"-"+ str(Date.day)
    data['date'] = name
    today = datetime.today()
    today = pd.to_datetime(today)
    print("Date", Date)
    data['NEW'] = pd.to_datetime(Date + pd.to_timedelta(data['remaining_days_sub'], unit='d'))
    print("today", today)
    data.loc[(today < pd.to_datetime(data['NEW'])) & (
        data['actual_Y'] == 'renew'), 'actual_Y'] = 'InProgress'

    data = data.loc[data['remaining_days_sub'].between(5,15)]

    features = data[['user_subscription_id', 'remaining_days_sub',
                     'date', 'cancelled_at','actual_Y', 'predicted_Y', 'cancel_prob', 'renew_prob']]
    features.to_csv(str(processed)+"CHURN_PREDICTION_UAE" + '.csv', encoding='utf-8', index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Date to compute predictions", required=True)
    parser.add_argument("--directory", help="PROJECT LOCATION",required=True)


    args = parser.parse_args()

    date = args.date
    directory = args.directory

    log_dir_path = os.path.join(directory, '..', 'logs')
    Path(log_dir_path).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        handlers = [ logging.FileHandler(Path(log_dir_path).joinpath('making_predictions'+'.log'),mode='w+') ]
    )

    logger = logging.getLogger(__name__)

    print(date)
    print(directory)
    logger.info("Directory:   %s" ,directory)

    prediction(Date=date,directory=directory)
