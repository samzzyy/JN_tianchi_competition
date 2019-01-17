import pandas as pd
import matplotlib.pyplot as plt

def plt_plot(y_real):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(len(y_real)),y_real, color='b')
    plt.show()

def hms2s(t):
    # print(t)
    h,m,s = t.strip().split(":")
    # print(h, m,s)
    # return int(h) * 3600 + int(m) * 60 + int(s)
    return int(h) * 60 + int(m)
def hm2m(t):
    # print(t)
    h, m = t.strip().split(":")
    # print(h, m)
    return int(h) * 60 + int(m)

def data_clean():
    df_temp=pd.read_csv("E:\\jinnan\\jinnan_round1_testA_20181227.csv", header=0,names=['id', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10',
           'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20',
           'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'B1', 'B2',
           'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13',
           'B14'],encoding = 'gb18030')
    hms_list=["B7","A16","A14","A11","A9","A5"]
    hms_hms_list=["B9","B4","A28"]

    hms_nan_list=["A7","B5","A26","A24"]
    hms_hms_nan_list=["B10","B11","A20",]
    df_temp=df_temp.fillna(-1)
    # where_is_nan=np.where(pd.isna(df_train))
    # print(where_is_nan)

    for hms_nan_column in hms_nan_list:
        print("______________________________________",hms_nan_column)
        df_temp.loc[df_temp[hms_nan_column]!=-1,hms_nan_column]=df_temp[hms_nan_column][df_temp[hms_nan_column]!=-1].apply(lambda x: hms2s(x))
    for hms_hms_nan_column in hms_hms_nan_list:
        print("______________________________________",hms_hms_nan_column)
        df_temp[hms_hms_nan_column + "range"] =-1
        df_temp.loc[df_temp[hms_hms_nan_column] != -1,hms_hms_nan_column + "range"] = \
            df_temp[hms_hms_nan_column][df_temp[hms_hms_nan_column] != -1].\
            apply(lambda x: hm2m(x.strip().split('-')[1]) - hm2m(x.strip().split('-')[0]))
        df_temp.loc[df_temp[hms_hms_nan_column] != -1, hms_hms_nan_column]= \
            df_temp[hms_hms_nan_column][df_temp[hms_hms_nan_column] != -1].\
                apply(lambda x: hm2m(x.strip().split('-')[0]))
    for hms_column in hms_list:
        print("______________________________________",hms_column)
        df_temp[hms_column]=df_temp[hms_column].apply(lambda x: hms2s(x))
    for hms_hms_column in hms_hms_list:
        print("______________________________________",hms_hms_column)
        df_temp[hms_hms_column+"range"] = df_temp[hms_hms_column].apply(lambda x: hm2m(x.strip().split('-')[1])-hm2m(x.strip().split('-')[0]))
        df_temp[hms_hms_column]=df_temp[hms_hms_column].apply(lambda x: hm2m(x.strip().split('-')[0]))

    df_temp.to_csv("E:\\jinnan\\test_after_clean.csv",index=False)

# data_clean()
# exit()

from sklearn import preprocessing
from sklearn.svm import SVR

df_train=pd.read_csv("E:\\jinnan\\train_after_clean.csv")
df_train=df_train[df_train['yield_rate']>=0.87]
df_train.drop(['B3', 'B13', 'A13', 'A18', 'A23'], axis=1, inplace=True)
for col in df_train.columns:
    rate = df_train[col].value_counts(normalize=True, dropna=False).values[0]
    if rate > 0.9:
        df_train.drop(columns=[col], axis=1, inplace=True)
print(df_train.columns,df_train.shape)
label_array=df_train["yield_rate"].values
df_train=df_train.drop(columns=['id','yield_rate'])
train_array=df_train.values
min_max_scaler = preprocessing.MinMaxScaler()
train_array = min_max_scaler.fit_transform(train_array)

clf=SVR()
print(clf)
clf.fit(train_array, label_array)
print("fit finished")

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

y_predicted=clf.predict(test_array)

df_predicted=pd.DataFrame({"id":df_test["id"].values,"predicted":y_predicted})
df_predicted.loc[:,"predicted"]=df_predicted["predicted"].apply(lambda x: round(x,3))
df_predicted.to_csv("E:\\jinnan\\predict_result\\jinnan_round1_submit_20181227.csv",header=None,index=False)
print(df_predicted)
plt_plot(y_predicted)


