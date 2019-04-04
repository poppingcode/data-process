# coding = utf-8
# the question1
import pandas as pd
import numpy as np
import time
import csv


file_read = "./data/raw/nyc_taxi_data.csv"
file_save = './data/raw/nyc_taxi_grid_data.csv'
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def read_csv_file(file_read_):
    return pd.read_csv(file_read_, dtype={
        'passengers': np.int,
        'up_y': np.float,
        'up_x':np.float,
        'off_y': np.float,
        'off_x':np.float},usecols = [0,1,2,3,4])

def data_process(df_,y,x):
    return df_['passengers'].groupby(
        [df_[y].map(lambda y_: int(y_/0.03125)),
         df_[x].map(lambda x_: int(x_/0.03125))
         ]).sum()

def calculate_result_save(up_cnt_,off_cnt_,file_save_):
    # calculate_result
    result = [[i+1,0,0] for i in range(1024)]
    for _,row in up_cnt_.iterrows():
        result[(1024-32*row['up_y']-32+row['up_x'])][1] = row['up_sum']
    for _,row in off_cnt_.iterrows():
        result[(1024-32*row['off_y']-32+row['off_x'])][2] = row['off_sum']
    birth_header = ["grid_index","up_passengers","off_passengers"]
    # save to file.csv
    with open(file_save_, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows([birth_header])
        writer.writerows(result)
        f.close()

if __name__ == '__main__':

    time_start=time.time()
    # read file
    df=read_csv_file(file_read)
    # up_cnt
    up_cnt = data_process(df,'up_y','up_x')
    print(up_cnt)
    up_cnt = pd.DataFrame({'up_sum':up_cnt}).reset_index()
    print(up_cnt)
    # off_cnt
    off_cnt = data_process(df,'off_y','off_x')
    off_cnt = pd.DataFrame({'off_sum':off_cnt}).reset_index()
    # save_result
    calculate_result_save(up_cnt,off_cnt,file_save)
    # print(off_cnt.head())
    time_end=time.time()
    print('totally cost',time_end-time_start)
    # print(off_cnt)
