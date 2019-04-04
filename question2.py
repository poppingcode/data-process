# coding = utf-8
# the question2
import pandas as pd
import numpy as np
import time
import csv
from surprise import Dataset
from surprise import Reader
from surprise import SVDpp
from surprise import accuracy
from surprise.model_selection import GridSearchCV

file_read = "./data/raw/pm25.csv"
file_save = './data/raw/pm25_predicted.csv'
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def read_csv_file(file_read_):
    return pd.read_csv(file_read_, dtype={
        'index': np.int,
        'PM2.5': np.str,
    },usecols = [0,1]).fillna(0)

def data_process(df_,x,y):  # 可以不正则化
    df_[x] = df_['index'].map(lambda y_: str(int(y_%72)))
    df_[y] = df_['index'].map(lambda x_: str(int(x_/72.00001)+1))
    df_ = df_.reindex(columns=["x","y","PM2.5"])
    df_.loc[df_.x == '0','x'] = 72
    return df_[~df_['y'].isin(['119'])]


def storage_zero_index_f(df):
    storage_zero_index_=[]
    zero_cnt_ =0
    for index,row in df.iterrows():
        if row['PM2.5'] ==0:
            storage_zero_index_.append([index+1, row['x'],row['y'],row['PM2.5']])
            zero_cnt_ = zero_cnt_+1
    return  storage_zero_index_,zero_cnt_

def Prediction(df,storage_zero_index_):
    result_ = []
    reader = Reader(rating_scale=(0.0,0.621794872),line_format='user item rating', sep='\t')
    data = Dataset.load_from_df(df,reader=reader)

    algo = SVDpp(n_epochs=50,reg_all=0.05,lr_all=0.005)
    trainset = data.build_full_trainset()
    algo.fit(trainset)
    testset = trainset.build_testset()
    predictions = algo.test(testset)
    # RMSE print
    accuracy.rmse(predictions, verbose=True)
    last_value = 0.0
    for tag in storage_zero_index_:
        pred = algo.predict(tag[1], tag[2], verbose=False)
        if (pred[3]!=0.0):
            result_.append([tag[0],pred[3]])
            last_value = pred[3]
        else:
            result_.append([tag[0],last_value])

    return result_

def Search_best_params(data):
    param_grid = {'n_epochs': [20,30,40], 'lr_all': [0.006,0.008,0.01],'reg_all': [0.2,0.3,0.4]}
    gs = GridSearchCV(SVDpp, param_grid, measures=['rmse'])
    gs.fit(data)
    # best RMSE score
    print(gs.best_score['rmse'])
    # # combination of parameters that gave the best RMSE score
    print(gs.best_params['rmse'])
    algo = gs.best_estimator['rmse']
    return algo,gs.best_params['rmse']

def calculate_result_save(file_save_,result_):
    # calculate_result
    birth_header = ["index","pm25"]
    # save to file.csv
    with open(file_save_, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows([birth_header])
        writer.writerows(result_)
        f.close()

if __name__ == '__main__':

    time_start=time.time()
    # read file
    df = read_csv_file(file_read)
    # data_process
    df = data_process(df,'x','y')
    # storage data to List storage_zero_index
    storage_zero_index,zero_cnt = storage_zero_index_f(df)

    # use the best_params to  def calculate_result_save(),the next line can be notes after get best_params
    # algo,best_params = Search_best_params(data)

    # # Prediction Value by SVD++
    result = Prediction(df,storage_zero_index)
    # save result
    calculate_result_save(file_save,result)

    time_end1=time.time()
    print('totally cost',time_end1-time_start)
