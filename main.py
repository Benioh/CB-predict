import data
import model    
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import random
import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score


best_model_path = 'best_model.pth'
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='CB emission prediction')
parser.add_argument('--model', type=str, default='CNN_LSTM', help='model for CB emission prediction')
args=parser.parse_args()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())


torch.manual_seed(3407)
np.random.seed(42)
random.seed(42)



train_x, train_y, test_x, test_y, scaler = data.load_data()
train_x1 = torch.Tensor(train_x).to(device)
train_y1 = torch.Tensor(train_y).to(device)
test_x1  = torch.Tensor(test_x).to(device)
test_y1  = torch.Tensor(test_y).to(device)

x_train,y_train=train_x,train_y

input_size  = 1  # 输入特征维度
conv_input  = 12
hidden_size = 64  # LSTM隐藏状态维度
num_layers  = 6  # LSTM层数
output_size = 1  # 输出维度（预测目标维度）

ninp = 512
nlayers=2




if args.model == 'LSTM':
   model      =  model.LSTM(input_size, hidden_size, num_layers, output_size)
   print("使用LSTM")
elif args.model == 'CNN_LSTM': 
   model      =  model.CNN_LSTM(conv_input,input_size, hidden_size, num_layers, output_size)
   print("使用CNN_LSTM")
elif args.model == 'Transformer':
     model      =  model.CNN_LSTM_Attention(conv_input,input_size, hidden_size, num_layers, output_size)
     print("使用Transformer")




model      =  model.to(device)

num_epochs = 1000
batch_size = 16#一次训练的数量
#优化器
optimizer = optim.Adam(model.parameters(),lr=0.0001,betas=(0.5,0.999))

#损失函数test
criterion=nn.MSELoss().to(device)

train_losses = []
test_losses  = []



def evaluate(test_x1, test_y1, model, scaler, device): 
    model.eval()
    with torch.no_grad(): 
        test_pred = model(test_x1.to(device)).detach().cpu().numpy()
    pred_y    = scaler.inverse_transform(test_pred).T[0]
    true_y = scaler.inverse_transform(test_y1.cpu().numpy()).T[0]
    
    # 计算指标
    mse_val  = mean_squared_error(true_y, pred_y)
    rmse_val = np.sqrt(mse_val)
    
    # 计算 R^2
    r2_val = r2_score(true_y, pred_y)
    
    # 计算 MAPE
    mape_val = np.mean(np.abs((pred_y - true_y) / true_y)) * 10
    
    print(f"MAPE: {mape_val:.2f}%")
    print(f"RMSE: {rmse_val}")
    print(f"R^2: {r2_val}")
    
    return true_y, pred_y


def train(): 
    global best_test_loss
    best_test_loss = float('inf')
    for epoch in range(num_epochs): 
        # 打乱数据
        permutation = torch.randperm(train_x1.size()[0])
     

        for i in range(0, train_x1.size()[0], batch_size):
            optimizer.zero_grad()

            indices = permutation[i:i + batch_size]
            batch_x, batch_y = train_x1[indices], train_y1[indices]
            # 训练
            model.train()
            output = model(batch_x)
            train_loss = criterion(output, batch_y)

            train_loss.backward()
            optimizer.step()

        # 每 50 个 epoch 打印一次训练损失和测试损失
        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                output    = model(test_x1)
                test_loss = criterion(output, test_y1)
                test_loss = test_loss.item()  # 转换为 Python 数值
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(model.state_dict(), 'best_model.pth')  # 保存模型参数

            train_losses.append(train_loss.item())
            test_losses.append(test_loss)
            print(f"epoch: {epoch}, train_loss: {train_loss}, test_loss: {test_loss}")

            # 评估模型并计算指标
            true_y, pred_y = evaluate(test_x1, test_y1, model, scaler, device)
           
           

train()


# def mse_cul(train_x1,train_y1,test_x1,test_y1,model,scaler): 
#     def mse(pred_y,true_y): 
#        return np.mean((pred_y-true_y) ** 2)
#     train_pred = model(train_x1).detach().numpy()
#     test_pred  = model(test_x1).detach().numpy()
#     pred_y     = np.concatenate((train_pred,test_pred))
#     pred_y     = scaler.inverse_transform(pred_y).T[0]
#     true_y     = np.concatenate((y_train,test_y))
#     true_y     = scaler.inverse_transform(true_y).T[0]
#     print(f"mse(pred_y,true_y):{mse(pred_y,true_y)}")
#     return true_y,pred_y

# def plot(true_y,pred_y): 
#     plt.figure()
#     plt.title("CNN_LSTM")
#     x = [i for i in range(len(true_y))]
#     plt.plot(x,pred_y,marker="o",markersize=1,label="pred_y")
#     plt.plot(x,true_y,marker="x",markersize=1,label="true_y")
#     plt.legend()
#     plt.show()






def plot(true_y, pred_y): 
    plt.figure()
    plt.title("CNN_LSTM")
    x = [i for i in range(len(pred_y))]
    plt.plot(x, true_y, marker="x", markersize=1, label="True Y")
    plt.plot(x, pred_y, marker="o", markersize=1, label="Predicted Y")
    plt.legend()
    plt.show()

# 使用 evaluate 函数来评估模型
true_y, pred_y = evaluate(test_x1, test_y1, model, scaler, device)
plot(true_y, pred_y)











# def mse_cul(test_x1, test_y1, model, scaler): 
#     def mse(pred_y, true_y)                     : 
#        return np.mean((pred_y - true_y) ** 2)
#     test_pred = model(test_x1.to(device)).detach().cpu().numpy()
#     pred_y    = scaler.inverse_transform(test_pred).T[0]
#     true_y    = scaler.inverse_transform(test_y1.cpu().numpy()).T[0]
#     print(f"mse(pred_y, true_y): {mse(pred_y, true_y)}")
#     return true_y, pred_y

# def plot(true_y,pred_y): 
#     plt.figure()
#     plt.title("CNN_LSTM")
#     x = [i for i in range(len(pred_y))]
#     plt.plot(x,true_y,marker="x",markersize=1,label="true_y")
#     plt.plot(x, pred_y, marker="o", markersize=1, label="pred_y")
#     plt.legend()
#     plt.show()


# true_y, pred_y = mse_cul(test_x1.to(device), test_y1.to(device), model, scaler)
# plot(true_y,pred_y)



