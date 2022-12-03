import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels as sm
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.model_selection import learning_curve, cross_val_score, train_test_split, cross_val_predict, cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, precision_score, plot_confusion_matrix, recall_score
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier, SGDClassifier, ElasticNet
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBRFClassifier, XGBClassifier, plot_importance
from imblearn.combine import SMOTEENN
from pprint import pprint
from sklearn.neighbors import KNeighborsClassifier
from imblearn.metrics import classification_report_imbalanced, geometric_mean_score
from imblearn.pipeline import make_pipeline
from imblearn.ensemble import RUSBoostClassifier, BalancedRandomForestClassifier, BalancedBaggingClassifier, EasyEnsembleClassifier
import os
from sklearn.naive_bayes import MultinomialNB
from warnings import filterwarnings
from keras import Sequential, layers, Input, callbacks, utils, regularizers
import tensorflow as tf

plt.style.use('fivethirtyeight')
sns.set_style('whitegrid')
filterwarnings('ignore')

maintenance = pd.read_csv("data_train/predictive_maintenance.csv")

# maintenance.head()
# print(maintenance.info())

"""COUNT"""
# plt.figure(figsize=(15,8))
# g=sns.countplot(x='Target', data=maintenance)
# for i, u in enumerate(maintenance['Target'].value_counts().values):
#     g.text(i, u, str(u), bbox=dict(facecolor='green', alpha=0.5))
# plt.title('Machine maintenance decision.')
# plt.show()

maintenance.drop(columns=['UDI'], inplace=True)
maintenance.describe()
maintenance.corr()
# print(maintenance.corr())
maintenance.skew()
maintenance.kurtosis()

"""Machine Failure Type"""
# plt.figure(figsize=(15,5))
# machine_failure = maintenance[maintenance['Failure Type'] != 'No Failure']
# hg=sns.countplot(x='Failure Type', data=machine_failure, order=['Heat Dissipation Failure', 'Power Failure',
#                                                               'Overstrain Failure', 'Tool Wear Failure',
#                                                               'Random Failures'])
# for i, u in enumerate(machine_failure['Failure Type'].value_counts().values):
#     hg.text(i, u, str(u), bbox=dict(facecolor='green', alpha=0.5))
# plt.title('Machine Failure Type')
# plt.show()

"""Relation between feature"""
# plt.figure(figsize=(15,5))
# sns.scatterplot(x='Air temperature [K]', y='Process temperature [K]',
#             hue='Target', alpha=0.75, data=maintenance)
# plt.xlabel('Air temperature [K]')
# plt.ylabel('Process temperature [K]')
# plt.show()

"""Feature engineering"""

"""Add column"""
maintenance['Power [W]'] = maintenance['Torque [Nm]'] * (2 * np.pi * maintenance['Rotational speed [rpm]'] / 60.0)
maintenance['Overstrain [minNm]'] = maintenance['Torque [Nm]'] * maintenance['Tool wear [min]']
maintenance['Heat dissipation [rpminK]'] = abs(
    maintenance['Air temperature [K]'] - maintenance['Process temperature [K]']) * maintenance['Rotational speed [rpm]']
# print(maintenance.tail(3))

"""check correlation"""
# plt.figure(figsize=(10,8))
# sns.heatmap(maintenance.corr(), annot=True, center=0)
# plt.title('Machine maintenance correlation with new features')
# plt.show()

# maintenance[['Power [W]', 'Overstrain [minNm]', 'Heat dissipation [rpminK]']].describe()

"""Power vs Torch """
# plt.figure(figsize=(15, 5))
# sns.scatterplot(y='Power [W]', x='Torque [Nm]', hue='Failure Type', data=maintenance)
# plt.show()

"""Power vs Heat """
# plt.figure(figsize=(15, 5))
# sns.scatterplot(y='Power [W]', x='Heat dissipation [rpminK]', hue='Failure Type', data=maintenance)
# plt.show()

data = maintenance.drop(columns=['Product ID', 'Type', 'Failure Type'])
matcorr = data.corr()
vif = pd.Series(np.linalg.inv(matcorr.to_numpy()).diagonal(), index=data.columns, name='vif_factor')
# print(vif.reset_index())

cols_selected = ['Process temperature [K]', 'Torque [Nm]', 'Tool wear [min]']
# We compute new vif
matcorr_ = data[cols_selected].corr()
vif_ = pd.Series(np.linalg.inv(matcorr_.to_numpy()).diagonal(), index=matcorr_.columns, name='vif_factor')
# print(vif_.reset_index())

"""Data preparation"""
Xdata = data[cols_selected]  #
target = maintenance['Target']  #
xtrain, xtest, ytrain, ytest = train_test_split(Xdata, target, stratify=target, random_state=0, test_size=0.2)
# print(f'Xtrain shape: {xtrain.shape} ytrain shape: {ytrain.shape}.')

"""Modeling"""


def ensemble_sampler_learning(X, y):
    ens_learners = {'bagg': BalancedBaggingClassifier(base_estimator=ExtraTreeClassifier(), random_state=0,
                                                      n_jobs=-1),
                    'rus': RUSBoostClassifier(base_estimator=ExtraTreeClassifier(), random_state=0),
                    'rfc': BalancedRandomForestClassifier(random_state=0, n_jobs=-1),
                    'easy': EasyEnsembleClassifier(base_estimator=ExtraTreeClassifier(),
                                                   random_state=0, n_jobs=-1)}
    results = {}
    imb_results = {}

    X, Xvalid, y, yvalid = train_test_split(X, y, stratify=y, random_state=42, train_size=0.66)

    print("Cross validation")
    print('=====================================================================')
    for u in ens_learners.keys():
        model = make_pipeline(RobustScaler(), PCA(n_components=0.95), ens_learners[u])
        cv_results = cross_validate(model, X, y=y, cv=5, n_jobs=-1, scoring="roc_auc",
                                    return_train_score=True, return_estimator=True)
        print('Learner', u)
        print(f"Training roc_auc mean +/- std. dev.: "
              f"{cv_results['test_score'].mean():.3f} +/- "
              f"{cv_results['test_score'].std():.3f}")
        print('\n')

        auc = []
        score = []
        for foldid, cv_model in enumerate(cv_results['estimator']):
            ypred = cv_model.predict(Xvalid)
            auc.append(roc_auc_score(yvalid, ypred))
            score.append(geometric_mean_score(yvalid, ypred, average='binary'))

            results[u] = auc
            imb_results[u] = score

    return results, imb_results


# auc_result_sampler, results_sampler = ensemble_sampler_learning(xtrain.values, ytrain)
# for u in auc_result_sampler.keys():
#     print(f'{u}: auc = {np.mean(auc_result_sampler[u])} +/- {np.std(auc_result_sampler[u])}')
#
# for u in results_sampler.keys():
#     print(f'{u}: G-mean = {np.mean(results_sampler[u])} +/- {np.std(results_sampler[u])}')

"""Training (1)"""
pipe_imbalanced = Pipeline([('scaler', RobustScaler()),
                            ('BRFC', BalancedRandomForestClassifier(n_jobs=-1,
                                                                    random_state=0,
                                                                    criterion='gini', max_depth=5,
                                                                    max_features='sqrt'))])
pipe_imbalanced.fit(xtrain.values, y=ytrain)
# print(classification_report_imbalanced(ytrain, pipe_imbalanced.predict(xtrain)))
# print(f'roc_auc training: {roc_auc_score(ytrain, pipe_imbalanced.predict(xtrain))}')

ypred = pipe_imbalanced.predict(xtest.values)
# print(classification_report_imbalanced(ytest, ypred))
# print(f'roc_auc test: {roc_auc_score(ytest, pipe_imbalanced.predict(xtest))}')

"""Machine Failure Type"""
failure_data = maintenance[maintenance['Failure Type'] != 'No Failure']
fdata = failure_data[cols_selected]
ftarget = failure_data['Failure Type']
# print(f'failure data shape: {fdata.shape}. failure type target shape = {ftarget.shape}.')

# we convert
ftarget = ftarget.astype("category")
ftarget.cat.categories = [0, 1, 2, 3, 4]
ftarget = ftarget.astype('int')

# fdata.plot.box(subplots=True, figsize=(15,5))
# plt.show()

"""We split our data."""
fxtrain, fxtest, fytrain, fytest = train_test_split(fdata, ftarget, stratify=ftarget, random_state=0,
                                                    test_size=0.2)


def best_enSamplerLearner(X, y):
    ens_learners = {'bagg': BalancedBaggingClassifier(base_estimator=ExtraTreeClassifier(), random_state=0,
                                                      n_jobs=-1),
                    'rus': RUSBoostClassifier(base_estimator=ExtraTreeClassifier(), random_state=0),
                    'rfc': BalancedRandomForestClassifier(random_state=0, n_jobs=-1),
                    'easy': EasyEnsembleClassifier(base_estimator=ExtraTreeClassifier(),
                                                   random_state=0, n_jobs=-1)}
    # print("Cross validation")
    # print('=====================================================================')
    for u in ens_learners.keys():
        model = make_pipeline(RobustScaler(), ens_learners[u])
        cv_results = cross_validate(model, X, y=y, cv=10, n_jobs=-1,
                                    return_train_score=False, return_estimator=True)

        train_score = []
        for foldid, cv_model in enumerate(cv_results['estimator']):
            y_pred = cv_model.predict(X)
            train_score.append(geometric_mean_score(y, y_pred))
        # print(u)
        # print(f'cross validation G-mean = {np.mean(train_score)} +/- {np.std(train_score)}.')
        # print('\n')


best_enSamplerLearner(fxtrain, fytrain)

"""Training (2)"""
imbalanced_pipe_failure = Pipeline([('scaler', RobustScaler()),
                                    ('BRFC', BalancedRandomForestClassifier(n_jobs=-1, random_state=0,
                                                                            criterion='entropy', max_depth=5,
                                                                            max_features='auto', n_estimators=100))])
imbalanced_pipe_failure.fit(fxtrain, fytrain)
print(classification_report_imbalanced(fytrain, imbalanced_pipe_failure.predict(fxtrain)))

print(f'Test G-mean = {geometric_mean_score(fytest, imbalanced_pipe_failure.predict(fxtest))}')
print(classification_report_imbalanced(fytest, imbalanced_pipe_failure.predict(fxtest)))

"""Put together"""


def MPM_model_decision(input_data=None):
    """
    input_data: 1d-dimensional array data

    return Failure and Failure Type
    """
    #
    failure_type = sorted(failure_data['Failure Type'].unique().tolist())
    ypred = pipe_imbalanced.predict(input_data)
    prob = pipe_imbalanced.predict_proba(input_data)[0]
    if ypred == 0:
        return f"Decision = {'No Failure'} with probability = {prob[ypred][0]}"
    else:
        y_pred = imbalanced_pipe_failure.predict(input_data)[0]
        prob = imbalanced_pipe_failure.predict_proba(input_data)[0]
        return f'Decision = {failure_type[y_pred]} \nWith probability = {prob[y_pred]}'


# print(f'Consider we have data: {np.array([[309.1, 4.6, 143]])}; the information of one product.')

# result = MPM_model_decision(np.array([[309.1, 4.6, 143]]))
# print(result)



