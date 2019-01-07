import pandas as pd
import numpy as np

def hms2s(t):
    # print(t)
    h,m,s = t.strip().split(":")
    # print(h, m,s)
    return int(h) * 3600 + int(m) * 60 + int(s)
def hm2m(t):
    # print(t)
    h, m = t.strip().split(":")
    # print(h, m)
    return int(h) * 60 + int(m)

def train_data_clean():
    df_train=pd.read_csv("E:\\jinnan\\jinnan_round1_train_20181227.csv", header=0,names=['id', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10',
           'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20',
           'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'B1', 'B2',
           'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13',
           'B14', 'yeild_rate'],encoding = 'gb18030')
    hms_list=["B7","A16","A14","A11","A9","A5"]
    hms_hms_list=["B9","B4","A20","A28"]

    hms_nan_list=["A7","B5","A26","A24",]
    hms_hms_nan_list=["B10","B11",]
    df_train=df_train.fillna(-1)
    # where_is_nan=np.where(pd.isna(df_train))
    # print(where_is_nan)

    for hms_nan_column in hms_nan_list:
        print(hms_nan_column)
        df_train.loc[df_train[hms_nan_column]!=-1,hms_nan_column]=df_train[hms_nan_column][df_train[hms_nan_column]!=-1].apply(lambda x: hms2s(x))
    for hms_hms_nan_column in hms_hms_nan_list:
        print(hms_hms_nan_column)
        df_train[hms_hms_nan_column + "range"] =-1
        df_train.loc[df_train[hms_hms_nan_column] != -1,hms_hms_nan_column + "range"] = \
            df_train[hms_hms_nan_column][df_train[hms_hms_nan_column] != -1].\
            apply(lambda x: hm2m(x.strip().split('-')[1]) - hm2m(x.strip().split('-')[0]))
        df_train.loc[df_train[hms_hms_nan_column] != -1, hms_hms_nan_column]=\
            df_train[hms_hms_nan_column][df_train[hms_hms_nan_column] != -1].\
                apply(lambda x: hm2m(x.strip().split('-')[0]))
    for hms_column in hms_list:
        print(hms_column)
        df_train[hms_column]=df_train[hms_column].apply(lambda x: hms2s(x))
    for hms_hms_column in hms_hms_list:
        print(hms_hms_column)
        df_train[hms_hms_column+"range"] = df_train[hms_hms_column].apply(lambda x: hm2m(x.strip().split('-')[1])-hm2m(x.strip().split('-')[0]))
        df_train[hms_hms_column]=df_train[hms_hms_column].apply(lambda x: hm2m(x.strip().split('-')[0]))

    # df_train.to_csv("E:\\jinnan\\train_after_clean.csv",index=False)
# train_data_clean()
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

df_train=pd.read_csv("E:\\jinnan\\train_after_clean.csv")
print(df_train.columns,df_train.shape)
label_array=df_train["yeild_rate"].values
df_train=df_train.drop(columns=['id','yeild_rate'])
train_array=df_train.values
min_max_scaler = preprocessing.MinMaxScaler()
train_array = min_max_scaler.fit_transform(train_array)

X_train, X_test, y_train, y_test = train_test_split(train_array, label_array, test_size=0.1, random_state=0)

clf=SVR()
print(clf)
clf.fit(X_train, y_train)
print("fit finished")

y_predicted=clf.predict(X_test)

error=np.mean(np.sum((y_predicted-y_test)**2))
print(error)