# -*- coding: utf-8 -*-
import os
import pandas as pd

from panel_econometrics.models import FixedEffectPanelLogit
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

filepath = os.path.dirname(os.path.abspath(__file__))
    
df = pd.read_csv(filepath + '/dataset/jtrain.csv')
df.set_index(['fcode','year'], inplace=True)

train, test = train_test_split(df, test_size = 0.2, random_state=1995)

FE = FixedEffectPanelLogit()
FE.fit(train, 'hrsemp', verbose=True)

FE.summary()

test.dropna(inplace=True)
y_pred = FE.predict(test[['grant', 'employ', 'sales']])
print('Accuracy score: %s' % accuracy_score(test['hrsemp'], y_pred))

FE.plot_trace_estimators()