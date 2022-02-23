import numpy as np
import pandas as pd

#train data
train_data = np.zeros((10000,10))
train_data = np.random.uniform(low = 0, high = 1, size = train_data.shape)


df_train_data = pd.DataFrame(train_data)
df_train_label = df_train_data.rank(axis = 1)
train_label = np.array(df_train_label)

train_out1 = []
for i in range(10):
    train_out1.append(train_label)
train_out1 = np.array(train_out1)

train_out2 = []
one = np.ones((10000,10))
for i in range(10):
    train_out2.append((i+1)*one)
train_out2 = np.array(train_out2)

train_out3 = train_out1 - train_out2
train_out3 = np.where(train_out3 == 0,1,0)

train_out4 = train_out3[0]


np.save("E:\\quant_research\\train the rank of ten points\\RNN_point\\data\\train_data_10num.npy",train_data)
np.save("E:\\quant_research\\train the rank of ten points\\RNN_point\\data\\train_label_10num.npy",train_out4)

#test data
train_data = np.zeros((10000,10))
train_data = np.random.uniform(low = 0, high = 1, size = train_data.shape)

df_train_data = pd.DataFrame(train_data)
df_train_label = df_train_data.rank(axis = 1)
train_label = np.array(df_train_label)

train_out1 = []
for i in range(10):
    train_out1.append(train_label)
train_out1 = np.array(train_out1)

train_out2 = []
one = np.ones((10000,10))
for i in range(10):
    train_out2.append((i+1)*one)
train_out2 = np.array(train_out2)

train_out3 = train_out1 - train_out2
train_out3 = np.where(train_out3 == 0,1,0)

train_out4 = train_out3[0]

np.save("E:\\quant_research\\train the rank of ten points\\RNN_point\\data\\test_data_10num.npy",train_data)
np.save("E:\\quant_research\\train the rank of ten points\\RNN_point\\data\\test_label_10num.npy",train_out4)
