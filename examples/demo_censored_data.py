# -*- coding: utf-8 -*-
import os
import pandas as pd

from panel_econometrics.censored_data_models import TobitModel
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

filepath = os.path.dirname(os.path.abspath(__file__))
    
df = pd.read_csv(filepath + '/dataset/ess6.csv')

train, test = train_test_split(df, test_size = 0.2, random_state=1995)
test.dropna(inplace=True)

#1. Tobit I Model
TM = TobitModel()
TM.fit(train, 'hinctnta', verbose=True)
TM.summary()

y_pred = TM.predict(test[['male', 'agea', 'wrkac6m']])
print('Tobit I Model, accuracy score: %s' % accuracy_score(test['hinctnta'], y_pred))