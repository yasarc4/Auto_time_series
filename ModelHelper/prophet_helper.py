from fbprophet import Prophet
from random import randint
from utils.datahelper import get_train_test
from utils.modelhelper import get_changepoints
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.metrics import mean_squared_error,mean_absolute_error



class ProphetHelper:
    def __init__(self, df=None, n_splits = 1, splits=[], holidays = None, n_models = 1, changepoint_prior_scale=None,
                 seasonality_prior_scale = None, holidays_prior_scale = None, max_train_window = 800,
                 mcmc_samples = None, interval_width = 0.8, uncertainty_samples = None, transformation_object=None):
        '''
        Created by Yasar
        '''
        self._df = df
        self._row_count = len(df)
        self._max_train_window = min(max_train_window, self._row_count)
        self._splits = self.get_splits(splits, n_splits)
        self._holidays = holidays
        self._n_models = n_models
        self._changepoint_prior_scale = self.get_changepoint_prior_scale(n_models)
        self._seasonality_prior_scale = self.get_seasonality_prior_scale(n_models)
        self._holidays_prior_scale = self.get_holidays_prior_scale(n_models)
        self._mcmc_samples = self.get_mcmc_samples(n_models)
        self._uncertainty_samples = self.get_uncertainty_samples(n_models)
        self._changepoints = get_changepoints(df)
        self._changepoints = [self._df['ds'].iloc[i] for i in self._changepoints]
        self._transformation_object = transformation_object
        self._trains_transformed_and_tests_raw = self.get_train_test(self._splits)
        self._models = self.initialise_models()
        self._performance = {}

    def get_date_str(self, i):
        return dt.datetime(*map(int,i.split('-')))

    def initialise_models(self):
        models = []
        for i in range(self._n_models):
            start_date = self._splits[-1][0]
            threshold_date = self._splits[0][1]
            changepoints = [i for i in self._changepoints if i<threshold_date and i>start_date]
            print (changepoints)
            print (self._changepoints)
            print (start_date, threshold_date)
            print ('$'*140)
            model = {#'changepoints' : [i for i in self._changepoints if i<self._splits[0][1]],
                            'seasonality_prior_scale':self._seasonality_prior_scale[i],
                            'holidays_prior_scale':self._holidays_prior_scale[i],
                            'changepoint_prior_scale':self._changepoint_prior_scale[i],
                            'mcmc_samples':self._mcmc_samples[i],
                            'interval_width':0.8, 'uncertainty_samples':self._uncertainty_samples[i]}
            models.append(model)
        return models

    def get_models_performance(self):
        for train, test in self._trains_transformed_and_tests_raw:
            print (train[:5])
            start_date = np.min(test['ds'])
            end_date = np.max(test['ds'])
            prediction_period = str(start_date) + ' to ' + str(end_date)
            self._performance[prediction_period] = {}
            performance = {}
            actuals = np.array(test['y'])
            predictions = np.zeros((len(test), self._n_models))
            mapes = []
            for i,model_params in enumerate(self._models):
                model = Prophet(holidays=self._holidays, **model_params)
                model.fit(train)
                prediction = model.predict(test[['ds']])['yhat']
                if self._transformation_object != None:
                    prediction = self._transformation_object.inverse_transform(prediction)
                predictions[:,i]=np.maximum(0,prediction)
                performance['Model-'+str(i)] = self.get_error_metrics(actuals=actuals, predicted=prediction)
                mapes.append(performance['Model-'+str(i)]['mape'])
            #performance['Stacked Model'] = self.get_error_metrics(actuals, np.product(predictions**(1.0/self._n_models),axis=1))
            performance['Stacked Model'] = self.get_stacked_model_perf(actuals, predictions, mapes)
            self._performance[prediction_period] = performance
        print (self._performance)

    def get_stacked_model_perf(self, actuals, predictions, mapes):
        mapes = list(enumerate(mapes))
        mapes = sorted(mapes, key = lambda x: x[1])
        weights = [1/(1+i) for i in range(self._n_models)]
        sum_weights = sum(weights)
        weights = [k for i,k in sorted([(i,k/sum(weights)) for (i,j),k in zip(mapes,weights)])]
        predictions = predictions**weights
        err_metrics = self.get_error_metrics(actuals,np.product(predictions, axis=1))
        err_metrics['weights'] = weights
        return err_metrics

    def get_error_metrics(self, actuals, predicted):
        mse = mean_squared_error(y_pred = predicted, y_true=actuals)
        rmse = mse**0.5
        mae = mean_absolute_error(y_pred = predicted, y_true=actuals)
        mapes = abs(predicted-actuals)*100/actuals
        mape = mapes[(~np.isnan(mapes)) & (np.isfinite(mapes))].mean()
        return {'mse':mse, 'rmse': rmse, 'mae': mae, 'mape':mape}

    def get_train_test(self, splits):
        for split in splits:
            print (split)
            train, test = get_train_test(self._df,'ds', *split)
            print ('*'*100)
            print (train[:5])
            print ('#'*100)
            if self._transformation_object != None:
                train['y'] = self._transformation_object.transform(train['y'])
            yield train, test

    def get_changepoint_prior_scale(self, n_models):
        return np.linspace(0.05, 0.4, n_models)

    def get_seasonality_prior_scale(self, n_models):
        return np.linspace(7.5, 37.5, n_models)

    def get_holidays_prior_scale(self, n_models):
        return np.linspace(7.5, 37.5, n_models)

    def get_mcmc_samples(self, n_models):
        return [randint(0,10) for i in range(n_models)]

    def get_uncertainty_samples(self, n_models):
        return [int(i) for i in np.linspace(int(0.1*self._row_count), int(0.4*self._row_count), n_models)]

    def check_splits(self, splits):
        try:
            for split in splits:
                assert((type(split) == tuple or type(split) == list) and len(split)==3)
        except:
            raise Exception('Please enter the proper splits!')
        return splits

    def get_random_splits(self, n_splits):
        splits = []
        start_idx = max(0,self._row_count - self._max_train_window - n_splits*7 + 7)
        split_idx = self._row_count - n_splits*7
        end_idx = split_idx + 7
        for i in range(n_splits):
            splits.append((self._df[y].iloc[start],self._df[y].iloc[split],self._df[y].iloc[end]))
            start_idx += 7
            split_idx += 7
            end_idx += 7
        return splits

    def get_splits(self, splits, n_splits):
        if n_splits==None and splits==[]:
            raise Exception('Please enter the number of splits (or) split dates for validation!')
        elif len(splits)>0:
            return self.check_splits(splits)
        else:
            return self.get_random_splits(n_splits)
