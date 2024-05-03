import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer,MultiheadAttention
import torch.optim as optim
import random
import math
import torch.nn.functional as F

class LSTM(nn.Module)                                                 : 
    def __init__(self, input_size, hidden_size, num_layers, output_size): 
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 定义CNN+LSTM模型类
class CNN_LSTM(nn.Module)                                                        : 
    def __init__(self, conv_input,input_size, hidden_size, num_layers, output_size): 
        super(CNN_LSTM,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.conv        = nn.Conv1d(conv_input,conv_input,1)
        self.lstm        = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc          = nn.Linear(hidden_size, output_size)
    
    def forward(self, x): 
        x  = self.conv(x)
        h0 = torch.zeros(self.num_layers,x.size(0), self.hidden_size).to(x.device) # 初始化隐藏状态h0
        c0 = torch.zeros(self.num_layers,x.size(0), self.hidden_size).to(x.device) # 初始化记忆状态c0
        out, _  = self.lstm(x, (h0, c0))  # LSTM前向传播
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出作为预测结果
        return out



# # 定义Transformer模型类
# class PositionalEncoding(nn.Module)                 : 
#     def __init__(self, d_model=512, dropout=0.1, max_len=20): 
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         self.d_model = d_model
#         pe          = torch.zeros(max_len, d_model)
#         position    = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term    = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe          = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)
#     def forward(self, x): 
#     # 假设 x 的形状是 [batch_size, time_step, 1]，并且 d_model > 1
#     # 首先，将 x 重塑为 [batch_size * time_step, 1]
#      x = x.view(-1, 1)
    
#     # 然后，扩展位置编码以匹配 x 的形状
#      pe = self.pe[:x.size(0), :]
    
#     # 将位置编码添加到 x
#      x = x + pe
    
#     # 如果需要，将 x 重塑回原始的形状 [batch_size, time_step, d_model]
#      x = x.view(-1, self.d_model)
    
#     # 应用 dropout
#      x = self.dropout(x)
    
#      return x



# class TransformerModel(nn.Module): 
   

#     def __init__(self,  ninp, nlayers,time_step=12,nhead=2,  dropout=0.5,nhid=200): 
#         super(TransformerModel, self).__init__()
#         try: 
#             from torch.nn import TransformerEncoder, TransformerEncoderLayer
#         except: 
#             raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
#         self.model_type          = 'Transformer'
#         self.src_mask            = None
#         self.pos_encoder         = PositionalEncoding(ninp, dropout)
#         encoder_layers           = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
#         self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
#         self.encoder = nn.Linear(1, ninp)
#         self.ninp                = ninp
#         self.decoder             = nn.Linear(ninp, 1)
#         self.init_weights()

 

#     def init_weights(self):
#         initrange = 0.1
#         nn.init.uniform_(self.encoder.weight, -initrange, initrange)
#         nn.init.zeros_(self.decoder.bias)
#         nn.init.uniform_(self.decoder.weight, -initrange, initrange)

#     def forward(self, src, has_mask=False):
#         if has_mask:
#             device = src.device
#             if self.src_mask is None or self.src_mask.size(0) != len(src):
#                 mask = self._generate_square_subsequent_mask(len(src)).to(device)
#                 self.src_mask = mask
#         else:
#             self.src_mask = None

#         src = self.encoder(src) * math.sqrt(self.ninp)
#         src = self.pos_encoder(src)
#         output = self.transformer_encoder(src, self.src_mask)
#         output = self.decoder(output)
#         return output


class CNN_LSTM_Attention(nn.Module)                                                                               : 
    def   __init__(self, conv_input, input_size, hidden_size, num_layers, output_size, nhead=2, dropout=0.5, nhid=200): 
        super(CNN_LSTM_Attention, self).__init__()
        self.hidden_size         = hidden_size
        self.num_layers          = num_layers
        self.conv                = nn.Conv1d(conv_input, conv_input, 1)
        self.lstm                = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        encoder_layers           = TransformerEncoderLayer(hidden_size, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)  # 使用 num_layers 作为 Transformer 层数
        self.attention           = MultiheadAttention(hidden_size, nhead, dropout=dropout)
        self.fc                  = nn.Linear(hidden_size, output_size)
    
    def forward(self, x): 
            x = self.conv(x)
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))  # LSTM 前向传播

            transformer_output = self.transformer_encoder(out)
            out = self.fc(transformer_output[:, -1, :])
            return out