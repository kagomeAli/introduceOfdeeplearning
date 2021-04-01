#%%

import os
import random
import numpy as np
import platform
import time
import re
import pandas as pd

#%%

def getWifiData(filepath, file_type = 'train'):
    file = open(filepath,"r")
    path = filepath.split('/')
    
        
    row_type= ""
    last_type = ""
    cur_floor = path[-2]
    
    #
    x = 0
    y = 0
    
    floor = ''
    if file_type == 'test':
        x = '-'
        y = '-'
        floor = '-'
        floor_area = path[-1].split('.')[0]
        
    if file_type == 'train':
        floor = STR_REPLACE[cur_floor]
        file_name = path[-1].split('.')[0]
        floor_area = cur_floor + '_' + file_name

    data = []
    
    bssid_rssi = []
    
    market_name = ''
    market_id = ''

    for ind,line in enumerate(file.readlines()):
        line_list = line.split()
        
        cur_type = line_list[1]

        if 'SiteID' in cur_type:
            market_id = line_list[1].split(':')[1]
            market_name = line_list[2].split(':')[1]
            
        if cur_type == "TYPE_WAYPOINT":
            x = line_list[2]
            y = line_list[3]
    
        if cur_type == "TYPE_WIFI":
            try:
                rssi_time = line_list[0]
                bssid_rssi.append([line_list[3], line_list[4]])
            except Exception as e:
                print(line)
                print("data error!")
                
        if cur_type != last_type and last_type == "TYPE_WIFI":
            bssid_rssi_sorted = sorted(bssid_rssi,key=(lambda x:x[1]), reverse= True)
            
            bssid_rssi_sorted = np.array(bssid_rssi_sorted)
            
            rssi_data =  bssid_rssi_sorted[:, 1]
            rssi_data =  list(rssi_data[0:SAVE_LEN])
            
            bssid_data =  bssid_rssi_sorted[:, 0]
            bssid_data =  list(bssid_data[0:SAVE_LEN])
            
            #print(type(rssi_data))
            
            #print(bssid_data)
            
            bssid_rssi = []
            
            rssi_data = list(rssi_data + ['0'] * (SAVE_LEN - len(rssi_data)))
            bssid_data = list(bssid_data + ['0'] * (SAVE_LEN - len(bssid_data)))
            
    
            csv_name = market_id +'_'+ floor_area +'_'+ rssi_time
            data.append(bssid_data + rssi_data + [x, y, floor,cur_floor, market_id, market_name, csv_name])

            
        last_type = cur_type
    
    return data, market_id

#%%

def getFilePath(root_path, file_list=[], dir_list=[]):
    # 获取该目录下所有的文件名称和目录名称
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        # 获取目录或者文件的路径
        dir_file_path = os.path.join(root_path, dir_file)
        # 判断该路径为文件还是路径
        if os.path.isdir(dir_file_path) and not dir_file.startswith('.'):
            dir_list.append(dir_file_path)
            # 递归获取所有文件和目录的路径
            getFilePath(dir_file_path, file_list, dir_list)
        else:
            file_list.append(dir_file_path)
    return file_list, dir_list

#%%

def getLocData(path, type = 'train'):
    data = []
    file_list, dir_list = getFilePath(path, [], [])
    for ind, file in enumerate(file_list):
        wifi, name = getWifiData(file, 'train')
        data.extend(wifi)
    return data, name

#%%

def getDir(path, loc):
    # 获取该目录下所有的文件夹名称
    dir_paths = []
    dir_names = []
    
    cur_path = os.path.join(path, loc)
    dir_or_files = os.listdir(cur_path)
    
    for dir_file in dir_or_files:
        
        dir_file_path = os.path.join(cur_path, dir_file)
        # 判断该路径为文件还是路径
        if os.path.isdir(dir_file_path):
            dir_paths.append(dir_file_path)
            dir_names.append(dir_file)
            
    return dir_paths, dir_names

#%%

#公共参数
SAVE_LEN = 200
BSSID_FEATS = [f'bssid_{i}' for i in range(SAVE_LEN)]
RSSI_FEATS  = [f'rssi_{i}' for i in range(SAVE_LEN)]
SITE_MSG= ["x", "y", "floor",'floor2', "site_id", "site_name", 'csv_name']
ALL_MSG = BSSID_FEATS + RSSI_FEATS + SITE_MSG
STR_REPLACE = {'1F':0,'2F':1, '3F':2, '4F':3,'5F':4, '6F':5,'7F':6, '8F':7, '9F':8,'B':-1, 'B1':-1,
               'B2':-2, 'B3':-3, 'BF':-1, 'BM':0, 'F1':0, 'F10':9, 'F2':1, 'F3':2, 'F4':3, 'F5':4, 'F6':4,
               'F7':6, 'F8':7, 'F9':8, 'G':0, 'L1':0, 'L10':9, 'L11':10, 'L2':1, 'L3':2, 'L4':3, 'L5':4,
               'L6':5, 'L7':6, 'L8':7, 'L9':8, 'LG1':-1, 'LG2':-2, 'LM':0, 'M':0, 'P1':-1, 'P2':-2}

#%%

path_loc_list = ['train']

date = time.strftime("%Y%m%d_%H_%M", time.localtime())

pathInput = '/home/aoi3080/Yanqin/dataset/indoorlocation/input/'
pathOutput = '/home/aoi3080/Yanqin/dataset/indoorlocation/output/'
pathModel = '/home/aoi3080/Yanqin/dataset/indoorlocation/model/'
pathLoc = '/home/aoi3080/Yanqin/dataset/indoorlocation/dataset/'

if platform.node() == 'niyanqindeMacBook-Pro.local':
    
    pathInput = '/Users/niyanqin/PycharmProjects/LearningDL/dataset/indoorlocation/input/'
    pathOutput = '/Users/niyanqin/PycharmProjects/LearningDL/dataset/indoorlocation/output/'
    pathModel = '/Users/niyanqin/PycharmProjects/LearningDL/dataset/indoorlocation/model/'
    
    pathLoc = '/Users/niyanqin/PycharmProjects/LearningDL/dataset/indoorlocation/'



#%%

path_train = ["train"]

train_all = []

save_csv_train = False
for loc in path_train:
    dir_lists, dir_names = getDir(pathLoc ,loc)
for ind,market in enumerate(dir_lists):
    cur_market, name = getLocData(market, 'train')
    train_all.extend(cur_market)
    if save_csv_train:
        cur_file=pd.DataFrame(columns=ALL_MSG, data=cur_market)
        cur_file.to_csv(pathInput + 'train/' + name + '.csv', index=0)

#%%

path_test = 'test'

save_csv_test = False
test_all = []

file_list, dir_list = getFilePath(pathLoc + path_test, [], [])

for ind, file in enumerate(file_list):
    test_data, name = getWifiData(file, 'test')
    test_all.extend(test_data)
    if save_csv_test:
        cur_file=pd.DataFrame(columns=ALL_MSG, data=test_data)
        cur_file.to_csv(pathInput + 'test/' + name +  '_' + str(ind + 1) + '.csv', index =0)

#%%

random.shuffle(test_all)
random.shuffle(train_all)

#%%

test_file=pd.DataFrame(columns=ALL_MSG, data=test_all)
test_file.to_csv(pathInput + 'test_all.csv', index=0)

train_file=pd.DataFrame(columns=ALL_MSG, data=train_all)
train_file.to_csv(pathInput + 'train_all.csv', index=0)
#%%
