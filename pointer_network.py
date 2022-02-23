import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt


train_data = np.load("E:\\quant_research\\train the rank of ten points\\RNN_point\\data\\train_data_10num.npy")
train_aim = np.load("E:\\quant_research\\train the rank of ten points\\RNN_point\\data\\train_label_10num.npy")
#转化矩阵维度,(seq_len, batch, input_size)
train_data = train_data.reshape(train_data.shape[0],10,1)
train_data = train_data.swapaxes(0, 1)
train_data = torch.from_numpy(train_data).type(torch.FloatTensor)
train_aim = torch.from_numpy(train_aim).type(torch.FloatTensor)

#测试集正确率
test_data = np.load("E:\\quant_research\\train the rank of ten points\\RNN_point\\data\\test_data_10num.npy")
test_aim = np.load("E:\\quant_research\\train the rank of ten points\\RNN_point\\data\\test_label_10num.npy")
#转化矩阵维度,(seq_len, batch, input_size)
test_data = test_data.reshape(test_data.shape[0],10,1)
test_data = test_data.swapaxes(0, 1)
test_data = torch.from_numpy(test_data).type(torch.FloatTensor)
test_aim = torch.from_numpy(test_aim).type(torch.FloatTensor)

class Encoder(nn.Module):

    #此处input_size = 1, hidden_size = 10
    def __init__(self, input_size, hidden_size, batch_size, bidirectional=True):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.batch_size = batch_size
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=False, bidirectional=bidirectional)

    def forward(self, inputs, hidden):
        output, hidden = self.lstm(inputs, hidden)   #input.shape 应为 (seq_len, batch, input_size)
        return output, hidden

    def init_hidden(self):
        return (torch.zeros(1 + int(self.bidirectional), self.batch_size, self.hidden_size),
                torch.zeros(1 + int(self.bidirectional), self.batch_size, self.hidden_size))   #(num_layers * num_directions, batch, hidden_size)


class AttentionDecoder(nn.Module):

    #此处hidden_size = hidden_size*(1 + bidirectional), output_size = 10, vocab_size = 10
    def __init__(self, hidden_size, output_size, batch_size, vocab_size,seq_len):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        self.attn = nn.Linear(hidden_size + output_size + vocab_size, 1)    #encoder层的size + 输出层的size  使用本层的hidden(encoder)和上一层的output计算
        self.lstm = nn.LSTM(hidden_size + vocab_size,          #使用本层的hidden和上一层的vocab(final output)计算
                            output_size)  # if we are using embedding hidden_size should be added with embedding of vocab size
        self.final = nn.Linear(output_size, vocab_size)

    def init_hidden(self):
        return (torch.zeros(1, self.batch_size, self.output_size),
                torch.zeros(1, self.batch_size, self.output_size))

    #decoder_hidden 为 (1, batch_size, output_size);encoder_outputs 为 (seq_len, batch, hidden_size);input 为 (seq_len, batch, voval_size)
    def forward(self, decoder_hidden, encoder_outputs, input):
        seq = 0
        #在一个时刻的迭代信息
        weights= []
        i = 0
        output = torch.zeros(self.batch_size, self.vocab_size)
        for i in range(len(encoder_outputs)):
            weights.append(self.attn(torch.cat((decoder_hidden[0][:].squeeze(0),encoder_outputs[i],output), dim=1)))

        #计算attention 的权值向量
        normalized_weight = F.softmax(torch.cat(weights, 1), 1)
        normalized_weights = normalized_weight

        #计算得到更新的h
        attn_applied = torch.bmm(normalized_weight.unsqueeze(1),
                                 encoder_outputs.transpose(0,1))  #- # 计算两个tensor的矩阵乘法，torch.bmm(a,b),tensor a 的size为(b,h,w),tensor b的size为(b,w,h),注意两个tensor的维度必须为3

        input_lstm = torch.cat((attn_applied.transpose(0,1)[0], output),
                               dim=1)  # if we are using embedding, use embedding of input here instead

        output_, hidden = self.lstm(input_lstm.unsqueeze(0), decoder_hidden)

        output = self.final(output_[0]) #output 为（vocab_size, output_size）
        #output = self.final2(output)

        # hidden0 = hidden[0].transpose(0, 1).reshape(batch_size, 1, -1).transpose(0, 1)
        # hidden1 = hidden[1].transpose(0, 1).reshape(batch_size, 1, -1).transpose(0, 1)
        # decoder_hidden = (hidden0, hidden1)
        # decoder_hiddens = decoder_hidden


        out = F.softmax(output,1)

        return out


#根据上述class定义attention网络

seq_len = 10
input_size = 1
hidden_size = 2
batch_size = train_data.shape[1]
bidirectional = True
output_size = hidden_size * (1 + bidirectional)
vocal_size = 10

input = []
for i in range(10):
    m = np.ones((10000,10))*i
    input.append(m)
input = np.array(input)
input = torch.from_numpy(input).type(torch.FloatTensor)

class pointer_atten(nn.Module):
    #num_classes表示最终每个样本输出的特征数
    def __init__(self):
        super(pointer_atten, self).__init__()
        self.layer1 = Encoder(input_size = input_size,
                              hidden_size = hidden_size,
                              batch_size = batch_size,
                              bidirectional=True)
        self.layer2 = AttentionDecoder(
            hidden_size = hidden_size * (1 + bidirectional),
            output_size = output_size,
            batch_size = batch_size,
            vocab_size = vocal_size,
            seq_len = 1
        )

    def forward(self,x):
        #x表示train_data
        output, hidden = self.layer1.forward(x, self.layer1.init_hidden())
        hidden0 = hidden[0].transpose(0, 1).reshape(batch_size, 1, -1).transpose(0, 1)
        hidden1 = hidden[1].transpose(0, 1).reshape(batch_size, 1, -1).transpose(0, 1)
        decoder_hidden = (hidden0, hidden1)
        encoder_outputs = output
        last_output = self.layer2.forward(decoder_hidden, output, input)

        return last_output

#定义网络Net
Net = pointer_atten()


# 损失函数和优化器
learning_rate = 0.05
Loss = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(Net.parameters(), lr=learning_rate)

###########################################
# 训练模型
###########################################
loss_list = []
True_list = []
num_epochs = 10000
epoch = 10000
batch = train_aim.detach().numpy().size

Net.load_state_dict(torch.load('E:\\quant_research\\train the rank of ten points\\RNN_point\\net_10num\\net720.pkl'))

for epoch in range(1000):
    train_data = Variable(train_data,requires_grad=True)
    train_aim = Variable(train_aim,requires_grad=True)

    # Forward pass
    outputs = Net(train_data)
    loss = Loss(outputs, train_aim)
    loss_list.append(loss)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch) % 10 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'
               .format(epoch+1,num_epochs,loss.item()))

        #计算拟合正确率
        # 设置绝对误差为0.01
        is_not = outputs.detach().numpy() - train_aim.detach().numpy()
        is_not = np.where(is_not < -0.1, 10, is_not)
        is_not = np.where(is_not < 0.1, 1, 0)
        T_pre = np.nansum(is_not)
        True_rate = T_pre / batch
        True_list.append(True_rate)
        print('预测正确率:', True_rate)

    if epoch % 10 ==0:
    # 存储训练参数
        torch.save(Net.state_dict(), 'E:\\quant_research\\train the rank of ten points\\\RNN_point\\net_10num\\net{}.pkl'.format(epoch))

loss_array = np.array(loss_list)
true_array = np.array(True_list)
np.save('E:\\quant_research\\train the rank of ten points\\\RNN_point\\loss',loss_array)
np.save('E:\\quant_research\\train the rank of ten points\\\RNN_point\\true',true_array)


#画图
loss_array = np.load('E:\\quant_research\\train the rank of ten points\\\RNN_point\\loss.npy',allow_pickle=True)
true_array = np.load('E:\\quant_research\\train the rank of ten points\\\RNN_point\\true.npy')

#测试集以及训练集正确率与损失值
#测试集
outputs = Net(train_data)
loss = Loss(outputs, train_aim)
label = np.argmax(outputs.detach().numpy(),axis = 1)
label_aim = np.argmax(train_aim.detach().numpy(),axis = 1)
True_rate = np.sum(label == label_aim) / 10000
print('预测集损失值%.5f，预测正确率:%.5f'%(loss,True_rate))

#测试集
outputs = Net(test_data)
loss = Loss(outputs, test_aim)
label = np.argmax(outputs.detach().numpy(),axis = 1)
label_aim = np.argmax(test_aim.detach().numpy(),axis = 1)
True_rate = np.sum(label == label_aim) / 10000
print('测试集损失值%.5f，预测正确率:%.5f'%(loss,True_rate))


# import torch
# import torch.nn as nn
# from torch import optim
# import torch.nn.functional as F
# from torch.autograd import Variable
# import numpy as np
# import matplotlib.pyplot as plt
#
# train_data = np.load("E:\\quant_research\\train the rank of ten points\\RNN_point\\data\\train_data.npy")
# train_aim = np.load("E:\\quant_research\\train the rank of ten points\\RNN_point\\data\\train_label.npy")
# # 转化矩阵维度,(seq_len, batch, input_size)
# train_data = train_data.reshape(train_data.shape[0], 5, 1)
# train_data = train_data.swapaxes(0, 1)
# train_data = torch.from_numpy(train_data).type(torch.FloatTensor)
# train_aim = torch.from_numpy(train_aim).type(torch.FloatTensor)
#
#
# class Encoder(nn.Module):
#
#     # 此处input_size = 1, hidden_size = 10
#     def __init__(self, input_size, hidden_size, batch_size, bidirectional=True):
#         super(Encoder, self).__init__()
#         self.hidden_size = hidden_size
#         self.input_size = input_size
#         self.batch_size = batch_size
#         self.bidirectional = bidirectional
#
#         self.lstm = nn.LSTM(input_size, hidden_size, batch_first=False, bidirectional=bidirectional)
#
#     def forward(self, inputs, hidden):
#         output, hidden = self.lstm(inputs, hidden)  # input.shape 应为 (seq_len, batch, input_size)
#         return output, hidden
#
#     def init_hidden(self):
#         return (torch.zeros(1 + int(self.bidirectional), self.batch_size, self.hidden_size),
#                 torch.zeros(1 + int(self.bidirectional), self.batch_size,
#                             self.hidden_size))  # (num_layers * num_directions, batch, hidden_size)
#
#
# class AttentionDecoder(nn.Module):
#
#     # 此处hidden_size = hidden_size*(1 + bidirectional), output_size = 10, vocab_size = 10
#     def __init__(self, hidden_size, output_size, batch_size, vocab_size, seq_len):
#         super(AttentionDecoder, self).__init__()
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.batch_size = batch_size
#         self.seq_len = seq_len
#         self.vocab_size = vocab_size
#
#         self.attn = nn.Linear(hidden_size + output_size + vocab_size,
#                               1)  # encoder层的size + 输出层的size  使用本层的hidden(encoder)和上一层的output计算
#         self.lstm = nn.LSTM(hidden_size + vocab_size,  # 使用本层的hidden和上一层的vocab(final output)计算
#                             output_size)  # if we are using embedding hidden_size should be added with embedding of vocab size
#         self.final = nn.Linear(output_size, vocab_size)
#
#     def init_hidden(self):
#         return (torch.zeros(1, self.batch_size, self.output_size),
#                 torch.zeros(1, self.batch_size, self.output_size))
#
#     # decoder_hidden 为 (1, batch_size, output_size);encoder_outputs 为 (seq_len, batch, hidden_size);input 为 (seq_len, batch, voval_size)
#     def forward(self, decoder_hidden, encoder_outputs, input):
#         weights = {}
#         decoder_hiddens = {}
#         decoder_hiddens[0] = decoder_hidden
#         attn_applied = {}
#         output = []
#         normalized_weights = {}
#         output.append(torch.zeros(self.batch_size, self.vocab_size))
#         for seq in range(self.seq_len):
#             seq = 0
#             # 在一个时刻的迭代信息
#             weights[seq] = []
#             i = 0
#             for i in range(len(encoder_outputs)):
#                 weights[seq].append(self.attn(
#                     torch.cat((decoder_hiddens[seq][0][:].squeeze(0), encoder_outputs[i], output[seq]), dim=1)))
#
#             # 计算attention 的权值向量
#             normalized_weight = F.softmax(torch.cat(weights[seq], 1), 1)
#             normalized_weights[seq] = normalized_weight
#
#             # 计算得到更新的h
#             attn_applied[seq] = torch.bmm(normalized_weight.unsqueeze(1),
#                                           encoder_outputs.transpose(0,
#                                                                     1))  # - # 计算两个tensor的矩阵乘法，torch.bmm(a,b),tensor a 的size为(b,h,w),tensor b的size为(b,w,h),注意两个tensor的维度必须为3
#
#             input_lstm = torch.cat((attn_applied[seq].transpose(0, 1)[0], output[seq]),
#                                    dim=1)  # if we are using embedding, use embedding of input here instead
#
#             output_, hidden = self.lstm(input_lstm.unsqueeze(0), decoder_hiddens[seq])
#
#             output.append(self.final(output_[0]))  # output 为（vocab_size, output_size）
#
#             hidden0 = hidden[0].transpose(0, 1).reshape(batch_size, 1, -1).transpose(0, 1)
#             hidden1 = hidden[1].transpose(0, 1).reshape(batch_size, 1, -1).transpose(0, 1)
#             decoder_hidden = (hidden0, hidden1)
#             decoder_hiddens[seq + 1] = decoder_hidden
#
#         # 对输出的output作用softmax
#         last_out = []
#
#         for i in range(self.seq_len):
#             m = F.softmax(output[i], 1)
#             last_out.append(m)
#
#         return last_out
#
#
# # 根据上述class定义attention网络
#
# seq_len = 5
# input_size = 1
# hidden_size = 2
# batch_size = train_data.shape[1]
# bidirectional = True
# output_size = hidden_size * (1 + bidirectional)
# vocal_size = 5
#
# input = []
# for i in range(10):
#     m = np.ones((10000, 10)) * i
#     input.append(m)
# input = np.array(input)
# input = torch.from_numpy(input).type(torch.FloatTensor)
#
#
# class pointer_atten(nn.Module):
#     # num_classes表示最终每个样本输出的特征数
#     def __init__(self):
#         super(pointer_atten, self).__init__()
#         self.layer1 = Encoder(input_size=input_size,
#                               hidden_size=hidden_size,
#                               batch_size=batch_size,
#                               bidirectional=True)
#         self.layer2 = AttentionDecoder(
#             hidden_size=hidden_size * (1 + bidirectional),
#             output_size=output_size,
#             batch_size=batch_size,
#             vocab_size=vocal_size,
#             seq_len=seq_len
#         )
#
#     def forward(self, x):
#         # x表示train_data
#         output, hidden = self.layer1.forward(x, self.layer1.init_hidden())
#         hidden0 = hidden[0].transpose(0, 1).reshape(batch_size, 1, -1).transpose(0, 1)
#         hidden1 = hidden[1].transpose(0, 1).reshape(batch_size, 1, -1).transpose(0, 1)
#         decoder_hidden = (hidden0, hidden1)
#         encoder_outputs = output
#         last_output = self.layer2.forward(decoder_hidden, output, input)
#
#         return last_output
#
#
# # 定义网络Net
# Net = pointer_atten()
#
# # 损失函数和优化器
# learning_rate = 0.05
# Loss = nn.MSELoss(reduction='mean')
# optimizer = torch.optim.Adam(Net.parameters(), lr=learning_rate)
#
# ###########################################
# # 训练模型
# ###########################################
# loss_list = []
# True_list = []
# num_epochs = 4151
# epoch = 0
# batch = train_aim.detach().numpy().size
# Net.load_state_dict(torch.load('E:\\quant_research\\train the rank of ten points\\RNN_point\\net\\net5140.pkl'))
# for epoch in range(4150, num_epochs):
#     train_data = Variable(train_data, requires_grad=True)
#     train_aim = Variable(train_aim, requires_grad=True)
#
#     # Forward pass
#     outputs = Net(train_data)
#     loss = 0
#     for i in range(seq_len):
#         loss += Loss(outputs[i], train_aim[i])
#     loss_list.append(loss)
#
#     # Backward and optimize
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     if (epoch) % 10 == 0:
#         print('Epoch [{}/{}], Loss: {:.4f}'
#               .format(epoch + 1, num_epochs, loss.item()))
#
#         # 计算拟合正确率
#         out = []
#         for i in range(seq_len):
#             out.append(outputs[i].detach().numpy())
#         pre = np.array(out)
#         # 设置绝对误差为0.01
#         is_not = pre - train_aim.detach().numpy()
#         is_not = np.where(is_not < -0.01, 10, is_not)
#         is_not = np.where(is_not < 0.01, 1, 0)
#         T_pre = np.nansum(is_not)
#         True_rate = T_pre / batch
#         True_list.append(True_rate)
#         print('预测正确率:', True_rate)
#
#     if epoch % 10 == 0:
#         # 存储训练参数
#         torch.save(Net.state_dict(),
#                    'E:\\quant_research\\train the rank of ten points\\\RNN_point\\net\\net{}.pkl'.format(epoch))
#
#
#
