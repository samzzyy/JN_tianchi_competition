import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plt_plot(y_real):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(len(y_real)),y_real, color='b')
    plt.show()

from sklearn import preprocessing
from sklearn.model_selection import KFold, RepeatedKFold
import xgboost as xgb
from sklearn.metrics import mean_squared_error

df_train=pd.read_csv("E:\\jinnan\\train_after_clean.csv")
df_train=df_train[df_train['yeild_rate']>=0.87]
df_train.drop(['B3', 'B13', 'A13', 'A18', 'A23'], axis=1, inplace=True)
for col in df_train.columns:
    rate = df_train[col].value_counts(normalize=True, dropna=False).values[0]
    if rate > 0.9:
        df_train.drop(columns=[col], axis=1, inplace=True)
print(df_train.columns,df_train.shape)
label_array=df_train["yeild_rate"].values
df_train=df_train.drop(columns=['id','yeild_rate'])
train_array=df_train.values
min_max_scaler = preprocessing.MinMaxScaler()
train_array = min_max_scaler.fit_transform(train_array)

df_test=pd.read_csv("E:\\jinnan\\test_after_clean.csv")
df_test.drop(['B3', 'B13', 'A13', 'A18', 'A23'], axis=1, inplace=True)
for col in df_test.columns:
    rate = df_test[col].value_counts(normalize=True, dropna=False).values[0]
    if rate > 0.9:
        df_test.drop(columns=[col], axis=1, inplace=True)
print(df_test.columns,df_test.shape)
columns_list=list(df_test.columns)
columns_list.remove("id")
test_array=df_test[columns_list].values
min_max_scaler = preprocessing.MinMaxScaler()
test_array = min_max_scaler.fit_transform(test_array)

xgb_params = {'eta': 0.005, 'max_depth': 10, 'subsample': 0.8, 'colsample_bytree': 0.8,
          'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True, 'nthread': 4}
folds = KFold(n_splits=5, shuffle=True, random_state=2018)
oof_xgb = np.zeros(len(df_train))
predictions_xgb = np.zeros(len(df_test))
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_array, label_array)):
    print("fold n°{}".format(fold_ + 1))
    trn_data = xgb.DMatrix(train_array[trn_idx], label_array[trn_idx])
    val_data = xgb.DMatrix(train_array[val_idx], label_array[val_idx])

    watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
    clf = xgb.train(dtrain=trn_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200,
                    verbose_eval=100, params=xgb_params)
    oof_xgb[val_idx] = clf.predict(xgb.DMatrix(train_array[val_idx]), ntree_limit=clf.best_ntree_limit)
    predictions_xgb += clf.predict(xgb.DMatrix(test_array), ntree_limit=clf.best_ntree_limit) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_xgb, label_array)))

# y_predicted=clf.predict(test_array)
#
# df_predicted=pd.DataFrame({"id":df_test["id"].values,"predicted":y_predicted})
# df_predicted.loc[:,"predicted"]=df_predicted["predicted"].apply(lambda x: round(x,3))
# df_predicted.to_csv("E:\\jinnan\\predict_result\\jinnan_round1_submit_20181227.csv",header=None,index=False)
# print(df_predicted)
# plt_plot(y_predicted)


