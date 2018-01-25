import pandas as pd

def load_csv(path, max_rows = None, date_col='txn_dttm', ts_col='amt_atmcam'):
    df = pd.read_csv(path)
    df = df[[date_col, ts_col]]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    if max_rows==None:
        return df
    else:
        return df.iloc[-max_rows-df[date_col].iloc[-max_rows].hour:]
