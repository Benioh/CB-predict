import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import warnings#避免一些可以忽略的报错
import random
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')

# #设置随机种子

# torch.backends.cudnn.deterministic = True#将cudnn框架中的随机数生成器设为确定性模式
# torch.backends.cudnn.benchmark = False#关闭CuDNN框架的自动寻找最优卷积算法的功能，以避免不同的算法对结果产生影响
# torch.manual_seed(42)
# np.random.seed(42)
# random.seed(42)


def split_data(data,time_step=12): 
    x = []
    y = []
    for i in range(len(data)-time_step):
        x.append(data[i:i+time_step])
        y.append(data[i+time_step])
    return np.array(x), np.array(y)

    

# 划分训练集和测试集
def train_test_split(dataX,datay,shuffle=False,percentage=0.84): 

    if shuffle:
        random_num=[index for index in range(len(dataX))]
        np.random.shuffle(random_num)
        dataX=dataX[random_num]
        datay=datay[random_num]
    split_num=int(len(dataX)*percentage)
    train_X=dataX[:split_num]
    train_y=datay[:split_num]
    test_X=dataX[split_num:]
    test_y=datay[split_num:]
    return train_X, train_y, test_X, test_y

def load_data(): 
    train_df = pd.read_csv("/home/chen/study/CB predict/Carbon_emission.csv")
    print(f"数据量:{len(train_df)}")

    c_e = train_df['carbon_emission']
    plt.plot([i for i in range(len(c_e))],c_e)
    from sklearn.preprocessing import MinMaxScaler
    # 创建MinMaxScaler对象
    scaler = MinMaxScaler()
    # 将数据进行归一化
    c_e = scaler.fit_transform(np.array(c_e).reshape(-1, 1))
      
    dataX,datay = split_data(c_e,time_step=12)
    print(f"dataX.shape:{dataX.shape},datay.shape:{datay.shape}")
    train_X, train_y, test_X, test_y = train_test_split(dataX,datay,shuffle=False)
    return train_X, train_y, test_X, test_y,scaler  #注意这里要把scaler返回出去，因为后面要用到它


if __name__ == "__main__":
    load_data()
    


    