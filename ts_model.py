from changepoint import ThreshDetector
from utils.datahelper import aggregate_df
from utils.modelhelper import get_models
from utils.dataloader import load_csv
import os
import ModelHelper
from ModelHelper import *
from TransformHelper import *

all_csvs = ['ATM_8178_ALL.csv','ATM_8212_ALL.csv','ATM_5145_ALL.csv','ATM_5384_ALL.csv','ATM_5463_ALL.csv','ATM_8877_ALL.csv','ATM_8948_ALL.csv','ATM_8959_ALL.csv']
# atm_8178 = load_csv('datasets/ATM_8178_ALL.csv', max_rows=21000) # 16% x 12%
# atm_8212 = load_csv('datasets/ATM_8212_ALL.csv', max_rows=21000) # 12% x 21%
# atm_5145 = load_csv('datasets/ATM_5145_ALL.csv', max_rows=21000) # 148% x 26%
# atm_5384 = load_csv('datasets/ATM_5384_ALL.csv', max_rows=21000) # 30% x 18%     Done
# atm_5463 = load_csv('datasets/ATM_5463_ALL.csv', max_rows=21000)  # 9% x 30%
# atm_8877 = load_csv('datasets/ATM_8877_ALL.csv', max_rows=21000)  # 19% x 12%    Done
# atm_8948 = load_csv('datasets/ATM_8948_ALL.csv', max_rows=21000)  # 22.7% x 16%
# atm_8959 = load_csv('datasets/ATM_8959_ALL.csv', max_rows=21000)  # 11.3% x 12%    Done

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
