import pandas as pd
from copy import deepcopy
from collections.abc import Iterable

from sklearn.pipeline import Pipeline
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

from utils.s3_utils import load_from_s3, write_to_s3
from config.config import BUCKET

import logging

logger = logging.getLogger(__name__)


class Model:
    save_attrs = [
        'algorithm',
        'hyper_parameters',
        'selected_features',
        'fitted'
    ]
    
    def __init__(
        self,
        algorithm: str = None,
        hyper_parameters: dict = None,
        selected_features: list = None
    ) -> None:
        # Set attributes
        self.model = None
        self.algorithm = algorithm

        self.hyper_parameters = None
        if hyper_parameters is not None:
            self.hyper_parameters = self.prepare_hyper_parameters(deepcopy(hyper_parameters))

        self.selected_features = selected_features
            
        self.fitted = False
        
    def prepare_hyper_parameters(
        self,
        hyper_parameters: dict
    ):
        if self.algorithm == 'sarimax':
            # Order
            if 'order' not in hyper_parameters:
                order = (
                    hyper_parameters['p'], 
                    hyper_parameters['d'], 
                    hyper_parameters['q']
                )
            else:
                order = hyper_parameters['order']
            
            # Seasonal Order
            if 'seasonal_order' not in hyper_parameters:                
                seasonal_order = (
                    hyper_parameters['seasonal_P'], 
                    hyper_parameters['seasonal_D'],
                    hyper_parameters['seasonal_Q'], 
                    hyper_parameters['seasonal_S']
                )
            else:
                seasonal_order = hyper_parameters['seasonal_order']
            
            # Trend
            if 'trend' not in hyper_parameters:
                trend = hyper_parameters['sarimax.trend']
            else:
                trend = hyper_parameters['trend']
            
            hyper_parameters = {
                'order': order,  # (p, d, q)
                'seasonal_order': seasonal_order,  # (P, D, Q, S)
                'trend': trend, 
                'measurement_error': False,
                'time_varying_regression': False, 
                'mle_regression': True,
                'simple_differencing': False,
                'enforce_stationarity': False, 
                'enforce_invertibility': False,
                'hamilton_representation': False, 
                'concentrate_scale': False,
                'trend_offset': 1, 
                'use_exact_diffuse': False, 
                'dates': None
            }
            
        if self.algorithm == 'random_forest':
            names = list(hyper_parameters.keys()).copy()
            for param_name in names:
                if 'random_forest.' in param_name:
                    correct_name = param_name.replace('random_forest.', '')
                    hyper_parameters[correct_name] = hyper_parameters.pop(param_name)
            
            hyper_parameters.update(**{
                'n_jobs': -1,
                'random_state': 23111997
            })
        
        if self.algorithm == 'lightgbm':
            names = list(hyper_parameters.keys()).copy()
            for param_name in names:
                if 'lightgbm.' in param_name:
                    correct_name = param_name.replace('lightgbm.', '')
                    hyper_parameters[correct_name] = hyper_parameters.pop(param_name)
                
            hyper_parameters.update(**{
                "objective": 'regression',
                "importance_type": 'split',
                "random_state": 23111997,
                "verbose": -1,
                "n_jobs": -1
            })
                
        return hyper_parameters
                
    def build(
        self,
        train_target: pd.Series = None, 
        train_features: pd.DataFrame = None
    ):
        if self.algorithm == 'expo_smooth':
            self.model = ExponentialSmoothing(
                train_target,
                trend=self.hyper_parameters['expo_smooth.trend'], 
                damped_trend=self.hyper_parameters['damped_trend'], 
                seasonal=self.hyper_parameters['seasonal'], 
                seasonal_periods=self.hyper_parameters['seasonal_periods']
            )
            
        elif self.algorithm == 'sarimax':
            if self.model is not None:
                self.hyper_parameters['start_ar_lags'] = self.model.ar_lags
                self.hyper_parameters['start_ma_lags'] = self.model.ma_lags
            
            self.model = SARIMAX(
                endog=train_target.values.astype(float),
                exog=train_features.values.astype(float),
                **self.hyper_parameters
            )
            
        elif self.algorithm == 'random_forest':
            self.model = RandomForestRegressor(**self.hyper_parameters)
            
        elif self.algorithm == 'lightgbm':
            self.model = LGBMRegressor(**self.hyper_parameters)
        
        self.fitted = False
            
    def fit(
        self,
        train_target: pd.Series = None, 
        train_features: pd.DataFrame = None
    ):
        if self.algorithm == 'expo_smooth':
            self.model.fit()
            
        elif self.algorithm == 'sarimax':
            self.model = self.model.fit(
                disp=False, 
                maxiter=50
            )
        
        elif self.algorithm == 'random_forest':
            self.model.fit(
                train_features.values.astype(float), 
                train_target.values.astype(float)
            )
        
        elif self.algorithm == 'lightgbm':
            # if hasattr(self.model, '_n_classes'):
            #     delattr(self.model, '_n_classes')

            self.model.fit(
                train_features.values.astype(float), 
                train_target.values.astype(float)
            )
            
        self.fitted = True
            
    def predict(
        self,
        forecast_features: pd.DataFrame = None, 
        forecast_dates: Iterable = None
    ):
        if self.algorithm == 'expo_smooth':
            return self.model.predict(
                self.model.params,
                start=forecast_dates[0],
                end=forecast_dates[-1],
            )
        
        elif self.algorithm == 'sarimax':
            return self.model.forecast(
                steps=forecast_features.shape[0],
                exog=forecast_features.values.astype(float)
            )
        
        elif self.algorithm == 'random_forest':
            return self.model.predict(
                forecast_features.values.astype(float)
            )
        
        elif self.algorithm == 'lightgbm':
            return self.model.predict(
                forecast_features.values.astype(float)
            )
        
    def save(
        self,
        as_champion: bool = True
    ):
        save_attrs = {
            attr_name: attr_value for attr_name, attr_value in self.__dict__.items()
            if attr_name in self.save_attrs
        }
        
        # Define paths
        if as_champion:
            logger.info('Saving new champion.\n')

            model_path = f"{BUCKET}/models/champion/champion.pickle"
            attrs_path = f"{BUCKET}/models/champion/attrs.pickle"
            
            
        else:
            logger.info('Saving new challenger.\n')

            model_path = f"{BUCKET}/models/challenger/challenger.pickle"
            attrs_path = f"{BUCKET}/models/challenger/attrs.pickle"
                
        # Save self.model
        write_to_s3(
            asset=self.model,
            path=model_path
        )

        # Save attrs
        write_to_s3(
            asset=save_attrs,
            path=attrs_path
        )
            
    def load(
        self,
        champion: bool = True
    ):
        # Define paths
        if champion:
            logger.info('Loading champion.\n')            
            
            model_path = f"{BUCKET}/models/champion/champion.pickle"
            attrs_path = f"{BUCKET}/models/champion/attrs.pickle"
            
            
        else:
            logger.info('Loading challenger.\n')
            
            model_path = f"{BUCKET}/models/challenger/challenger.pickle"
            attrs_path = f"{BUCKET}/models/challenger/attrs.pickle"
                
        # Load self.model
        self.model = load_from_s3(path=model_path)

        # Load attrs
        attrs: dict = load_from_s3(path=attrs_path)
        
        for attr_name, attr_value in attrs.items():
            setattr(self, attr_name, attr_value)