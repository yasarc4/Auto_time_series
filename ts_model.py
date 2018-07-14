from changepoint import ThreshDetector
from utils.datahelper import aggregate_df
from utils.modelhelper import get_models
from utils.dataloader import load_csv
import os
import ModelHelper
from ModelHelper import *
from TransformHelper import *

all_csvs = ['data1.csv','data2.csv']

dataset = all_csvs[5]
dataset_home = os.environ['DATASET_HOME']


day_level_df_total = aggregate_df(load_csv(dataset_home+dataset))
print ('Predicting for the dataset: ',dataset)

threshold_dates = [('2013-11-01','2016-11-01','2016-11-07'),
                   ('2013-11-08','2016-11-08','2016-11-14'),
                   ('2013-11-15','2016-11-15','2016-11-21'),
                   ('2013-11-22','2016-11-22','2016-11-28')]
# get_models(day_level_df_total, threshold_dates)

prophet_data = day_level_df_total
prophet_data.columns=['ds','y']
print (prophet_data[:5])
# prophet_model = ProphetHelper(df= prophet_data, splits = [('2013-11-01','2016-11-01','2016-11-07')],n_models=5)
prophet_model = ProphetHelper(df= prophet_data, splits = threshold_dates,n_models=5, transformation_object=MinMaxTransformer())

prophet_model.get_models_performance()
