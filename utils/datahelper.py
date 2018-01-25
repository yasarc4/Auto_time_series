import pandas as pd
import numpy as np
import datetime as dt

def aggregate_df(df, aggregation_level='D',date_col=None, ts_col=None):
    if date_col==None and ts_col==None:
        date_col, ts_col = list(df.columns)
    # df[ts_col].loc[df[ts_col<0]]=0
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except:
        pass
    agg_levels = {'H': '%Y-%m-%d %H',
                  'D': '%Y-%m-%d',
                  'M': '%Y-%m',
                  'Y': '%Y'}
    df[aggregation_level] = df[date_col].apply(lambda x: x.strftime(agg_levels[aggregation_level]))
    agg_df = df.groupby(aggregation_level).agg({ts_col:np.mean}).reset_index()
    agg_df.columns = [date_col+'_aggregated_by_'+aggregation_level, ts_col+'_aggregated_by_'+aggregation_level]
    return agg_df

def get_train_test(df, date_col, start_date, split_date, end_date):

    try:
        return df[(df[date_col]>=start_date) & (df[date_col]<split_date)].reset_index(drop=True),df[(df[date_col]>=split_date) & (df[date_col]<=end_date)].reset_index(drop=True)
    except:
        print ('-'*30,'  IN EXCEPT  ','-'*30)
        start_date = dt.datetime(*map(int,start_date.split('-')))
        split_date = dt.datetime(*map(int,split_date.split('-')))
        end_date = dt.datetime(*map(int,end_date.split('-')))
        print (start_date, split_date, end_date)
        return (df[(pd.to_datetime(df[date_col])>=start_date) & (pd.to_datetime(df[date_col])<split_date)].reset_index(drop=True),
                df[(pd.to_datetime(df[date_col])>=split_date) & (pd.to_datetime(df[date_col])<=end_date)].reset_index(drop=True))


def get_week_of_month(x):
    d = int((x.day-1)/7)
    if d>3:
        return '3'
    else:
        return str(d)

def get_all_date_comps(df, date_col):
    date_series = pd.to_datetime(df[date_col])
    df['by_month'] = date_series.apply(lambda x: x.strftime('%b'))
    df['by_day_of_week'] = date_series.apply(lambda x: x.strftime('%w'))
    df['by_week_of_month'] = date_series.apply(lambda x: get_week_of_month(x))
    return df
