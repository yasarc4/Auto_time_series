import pandas as pd
import numpy as np
import datetime as dt
from sklearn.metrics import mean_squared_error,mean_absolute_error
from .datahelper import *
from changepoint import ThreshDetector

def get_models(day_level_df_total, threshold_dates):
    for start,split,end in threshold_dates:
        print('Start date = ',start,'\nSplit date = ',split,'\nEnd date = ',end)
        day_level_df,day_level_df_test = get_train_test(day_level_df_total, 'txn_dttm_aggregated_by_D',start,split,end)
        get_model(day_level_df, day_level_df_test)

def get_model(day_level_df, day_level_df_test):
    ## TO-DO: Modularise after finalising approach
    change_points = get_changepoints(day_level_df)
    day_level_df['rolling_mean']=pd.rolling_mean(day_level_df['amt_atmcam_aggregated_by_D'],7).shift(1)
    day_level_df['diff_from_prev_period_mean'] = day_level_df['amt_atmcam_aggregated_by_D']-day_level_df['rolling_mean']
    day_level_df['secs_since_epoch']=day_level_df['txn_dttm_aggregated_by_D'].apply(lambda x: float(dt.datetime.strptime(x,'%Y-%m-%d').strftime('%s')))
    day_level_df_test['secs_since_epoch']=day_level_df_test['txn_dttm_aggregated_by_D'].apply(lambda x: float(dt.datetime.strptime(x,'%Y-%m-%d').strftime('%s')))
    start = 7
    fitted = [np.nan]*7
    formulas = []
    for end in change_points+[len(day_level_df)]:
        data = day_level_df.iloc[start:end]
        formula = (np.polyfit(data['secs_since_epoch'].tolist(),data['diff_from_prev_period_mean'].tolist(),1))
        print(formula)
        formulas.append(formula)
        fitted = fitted + (data['secs_since_epoch']*formula[0] + formula[1]).tolist()
        start = end
    day_level_df_test['trend'] = formulas[-1][0]*day_level_df_test['secs_since_epoch'] + formulas[-1][1]
    day_level_df['trend'] = fitted
    day_level_df['diff_from_trend']= day_level_df['diff_from_prev_period_mean'] - day_level_df['trend']
    day_level_df = get_all_date_comps(day_level_df,'txn_dttm_aggregated_by_D')
    day_level_df_test = get_all_date_comps(day_level_df_test,'txn_dttm_aggregated_by_D')
    yearly_seasonality_by_month = day_level_df.groupby('by_month').agg({'diff_from_trend':np.mean}).reset_index()
    yearly_seasonality_by_month.columns=['by_month','yearly_seasonality_by_month']
    day_level_df=pd.merge(left=day_level_df,right=yearly_seasonality_by_month,on='by_month',how='left')
    day_level_df_test=pd.merge(left=day_level_df_test,right=yearly_seasonality_by_month,on='by_month',how='left')
    day_level_df['diff_from_yearly_seasonality'] = day_level_df['diff_from_trend'] - day_level_df['yearly_seasonality_by_month']
    monthly_seasonality_by_week = day_level_df.groupby('by_week_of_month').agg({'diff_from_yearly_seasonality':np.mean}).reset_index()
    monthly_seasonality_by_week.columns=['by_week_of_month','monthly_seasonality_by_week']
    day_level_df=pd.merge(left=day_level_df,right=monthly_seasonality_by_week,on='by_week_of_month',how='left')
    day_level_df['diff_from_monthly_seasonality'] = day_level_df['diff_from_yearly_seasonality'] - day_level_df['monthly_seasonality_by_week']
    day_level_df_test=pd.merge(left=day_level_df_test,right=monthly_seasonality_by_week,on='by_week_of_month',how='left')
    weekly_seasonality_by_day = day_level_df.groupby('by_day_of_week').agg({'diff_from_monthly_seasonality':np.mean}).reset_index()
    weekly_seasonality_by_day.columns=['by_day_of_week','weekly_seasonality_by_day']
    day_level_df=pd.merge(left=day_level_df,right=weekly_seasonality_by_day,on='by_day_of_week',how='left')
    day_level_df['diff_from_weekly_seasonality'] = day_level_df['diff_from_monthly_seasonality'] - day_level_df['weekly_seasonality_by_day']
    day_level_df_test=pd.merge(left=day_level_df_test,right=weekly_seasonality_by_day,on='by_day_of_week',how='left')
    day_level_df['expected_diffs'] = day_level_df['trend'] + day_level_df['yearly_seasonality_by_month'] + day_level_df['monthly_seasonality_by_week'] + day_level_df['weekly_seasonality_by_day']
    day_level_df_test['expected_diffs'] = day_level_df_test['trend'] + day_level_df_test['yearly_seasonality_by_month'] + day_level_df_test['monthly_seasonality_by_week'] + day_level_df_test['weekly_seasonality_by_day']
    day_level_df['error_metric'] = day_level_df['diff_from_prev_period_mean'] - day_level_df['expected_diffs']
    expected_values = day_level_df['amt_atmcam_aggregated_by_D'].iloc[:7].tolist()
    for i in day_level_df['expected_diffs'].iloc[7:]:
        prev_mean = np.mean(expected_values[-7:])
        expected_values.append(prev_mean+i)
    day_level_df['expected_vals'] = expected_values
    train_errors = get_error_metrics(day_level_df)
    print('Train Error : ',train_errors)
    expected_values_test = day_level_df['amt_atmcam_aggregated_by_D'].iloc[-7:].tolist()
    for i in day_level_df_test['expected_diffs']:
        prev_mean = np.mean(expected_values_test[-7:])
        expected_values_test.append(prev_mean+i)
    day_level_df_test['expected_vals'] = expected_values_test[7:]
    test_errors = get_error_metrics(day_level_df_test)
    print('Test Error : ',test_errors)


def get_error_metrics(df):
    mse = mean_squared_error(y_pred = df['expected_vals'], y_true=df['amt_atmcam_aggregated_by_D'])
    rmse = mse**0.5
    mae = mean_absolute_error(y_pred = df['expected_vals'], y_true=df['amt_atmcam_aggregated_by_D'])
    mapes = abs(df['expected_vals']-df['amt_atmcam_aggregated_by_D'])*100/df['amt_atmcam_aggregated_by_D'].reset_index(drop=True)
    mape = mapes[(~np.isnan(mapes)) & (np.isfinite(mapes))].mean()

    return {'mse':mse, 'rmse': rmse, 'mae': mae, 'mape':mape}

def get_changepoints(df, ts_col=None, agg_level='D'):
    thresholds = {'D':(0.25, 15, 42),
                  'W':(0.25, 8, 20),
                  'M':(0.25, 24, 24)}
    detector = ThreshDetector(*thresholds[agg_level])
    # df['change_point']=False
    if ts_col==None:
        ts_col = df.columns[1]
    change_points = []
    for i in df[ts_col]:
        detector.step(i)
    return detector.changepoints
