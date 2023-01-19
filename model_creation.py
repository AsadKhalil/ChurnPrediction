import pandas as pd
import numpy as np
import os
from datetime import date
import argparse
import logging
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import *

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
# save model to file
import pickle
from pathlib import Path

# check version number
import imblearn
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings("ignore")

seed = 7

parser = argparse.ArgumentParser()
parser.add_argument("--directory", help="PROJECT LOCATION",required=True)

args = parser.parse_args()

#Getting command line arguments
directory =args.directory


log_dir_path = os.path.join(directory, '..', 'logs')
Path(log_dir_path).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    handlers = [ logging.FileHandler(Path(log_dir_path).joinpath('model_creation'+'.log'),mode='w+') ]
)

logger = logging.getLogger(__name__)

logger.info("Directory:   %s" ,directory)

os.chdir(directory)
os.chdir('..')
print(os.getcwd())
#Setting Directories

training = os.getcwd()+'/training'

model_directory =os.getcwd()
Path(model_directory+"/model").mkdir(parents=True, exist_ok=True)
model_directory =model_directory+"/model/"

train = pd.read_csv('training/training.csv')

train = train[['last_order_rating','last_failed_order','last_order_OTIF','last_order_issue'
               ,'subscription_type','saving','delivered_orders','avg_rating','total_order_issues','payment_declined_orders',
            'total_cancel','no_order_days','remaining_days_sub','otif_late',
               'order_cancel','payment_fail','issue_count','actual_Y']].copy()
logger.info("Reading File:   %s")

train['last_order_rating'] = train['last_order_rating'].round(1)
train['avg_rating'] = train['avg_rating'].round(1)
train['saving'] = train['saving'].round(1)
train['otif_late'] = train['otif_late'].round(1)

train['otif_late'] = train['otif_late'] * 100
lbl = preprocessing.LabelEncoder()
train['subscription_type'] = lbl.fit_transform(train['subscription_type'].astype(str))

train.drop_duplicates(['last_order_rating','last_failed_order','last_order_OTIF','last_order_issue'
               ,'subscription_type','saving','delivered_orders','avg_rating','total_order_issues','payment_declined_orders',
            'total_cancel','no_order_days','remaining_days_sub','otif_late',
               'order_cancel','payment_fail','issue_count'],inplace=True)

train.drop_duplicates(['last_failed_order','last_order_OTIF','last_order_issue'
               ,'subscription_type','delivered_orders','avg_rating','total_order_issues','payment_declined_orders',
            'total_cancel','no_order_days','otif_late'],inplace=True)

X = train[['last_failed_order','last_order_OTIF','last_order_issue'
               ,'subscription_type','delivered_orders','avg_rating','total_order_issues','payment_declined_orders',
            'total_cancel','no_order_days','otif_late']]

Y = train[['actual_Y']]

under = RandomUnderSampler(sampling_strategy='majority')

# now to comine under sampling 
X_under, y_under = under.fit_resample(X, Y)

X_train, X_test, y_train, y_test = train_test_split(X_under, y_under, test_size=0.2, random_state=seed)


model = XGBClassifier()
model.fit(X_train, y_train)

logger.info("Saving Model")

pickle.dump(model, open(model_directory+'model.dat', "wb"))


##########################################################################################################################
##logging Results

##TRAINING CANCEL
logger.info("training Data Cancel Label Results")

y_pred = model.predict(X_train)
# predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_train, y_pred)

logger.info("Accuracy: %.2f%%" % (accuracy * 100.0))
logger.info("Precision: %.2f%%" % (precision_score(y_train, y_pred,average='binary',pos_label='cancel') * 100.0))
logger.info("Recall: %.2f%%" % (recall_score(y_train, y_pred,average='binary',pos_label='cancel')  * 100.0))
logger.info("F1-score: %.2f%%" % (f1_score(y_train, y_pred,average='binary',pos_label='cancel')  * 100.0))


logger.info("training Data Renew Label Results")

# make predictions for test data
y_pred = model.predict(X_train)
# predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_train, y_pred)


logger.info("Accuracy: %.2f%%" % (accuracy * 100.0))
logger.info("Precision: %.2f%%" % (precision_score(y_train, y_pred,average='binary',pos_label='cancel') * 100.0))
logger.info("Recall: %.2f%%" % (recall_score(y_train, y_pred,average='binary',pos_label='cancel')  * 100.0))
logger.info("F1-score: %.2f%%" % (f1_score(y_train, y_pred,average='binary',pos_label='cancel')  * 100.0))

logger.info("VALIDATION DATA CANCEL Label Results")

#VALIDATION CANCEL
# make predictions for test data
y_pred = model.predict(X_test)
# predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)

logger.info("Accuracy: %.2f%%" % (accuracy * 100.0))
logger.info("Precision: %.2f%%" % (precision_score(y_test, y_pred,average='binary',pos_label='cancel') * 100.0))
logger.info("Recall: %.2f%%" % (recall_score(y_test, y_pred,average='binary',pos_label='cancel')  * 100.0))
logger.info("F1-score: %.2f%%" % (f1_score(y_test, y_pred,average='binary',pos_label='cancel')  * 100.0))


#VALIDATION RENEW
logger.info("VALIDATION DATA RENEW Label Results")



# make predictions for test data
y_pred = model.predict(X_test)
# predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)

logger.info("Accuracy: %.2f%%" % (accuracy * 100.0))
logger.info("Precision: %.2f%%" % (precision_score(y_test, y_pred,average='binary',pos_label='cancel') * 100.0))
logger.info("Recall: %.2f%%" % (recall_score(y_test, y_pred,average='binary',pos_label='cancel')  * 100.0))
logger.info("F1-score: %.2f%%" % (f1_score(y_test, y_pred,average='binary',pos_label='cancel')  * 100.0))
