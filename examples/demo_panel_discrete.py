# -*- coding: utf-8 -*-
import os
import pandas as pd

from panel_econometrics.panel_discrete_models import FixedEffectPanelModel, RandomEffectsPanelModel
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

filepath = os.path.dirname(os.path.abspath(__file__))
    
df = pd.read_csv(filepath + '/dataset/jtrain.csv')
df.set_index(['fcode','year'], inplace=True)

df = df.head(12)
train = df.head(9)
test = df.ix[9:]

# train, test = train_test_split(df, test_size = 0.2, random_state=1995)
# test.dropna(inplace=True)


# #1. Fixed Effects Logit Model
# FE = FixedEffectPanelLogit()
# FE.fit(train, 'hrsemp', verbose=True)
# FE.summary()

# y_pred = FE.predict(test[['grant', 'employ', 'sales']])
# print('FE Logit, accuracy score: %s' % accuracy_score(test['hrsemp'], y_pred))

#2. Random Effects Probit Model
# RE = RandomEffectsPanelModel('probit')
# RE.fit(train, 'hrsemp', verbose=True)
# RE.summary()

# y_pred = RE.predict(test[['grant', 'employ', 'sales']])
# print('RE Probit, accuracy score: %s' % accuracy_score(test['hrsemp'], y_pred))


#3. Random Effects Logit Model
RE = RandomEffectsPanelModel('logit')
RE.fit(train, 'hrsemp', verbose=True)
RE.summary()

y_pred = RE.predict(test[['grant', 'employ', 'sales']])
print('RE Logit, accuracy score: %s' % accuracy_score(test['hrsemp'], y_pred))