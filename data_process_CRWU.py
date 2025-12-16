import random
import numpy as np
import scipy.io as scio
from sklearn import preprocessing

def open_data(bath_path,key_num):
    path = bath_path + str(key_num) + ".mat"
    str1 =  "X" + "%03d"%key_num + "_DE_time"
    data = scio.loadmat(path)
    data = data[str1]
    return data


def deal_data(data,length,label):
    data = np.reshape(data,(-1))
    num = len(data)//length
    data = data[0:num*length]

    data = np.reshape(data,(num,length))

    min_max_scaler = preprocessing.MinMaxScaler()

    data = min_max_scaler.fit_transform(np.transpose(data,[1,0]))
    data = np.transpose(data,[1,0])
    label = np.ones((num,1))*label
    return np.column_stack((data,label)) 


def split_data(data,split_rate):
    length = len(data)
    num1 = int(length*split_rate[0])
    num2 = int(length*split_rate[1])

    index1 = random.sample(range(num1),num1)
    train = data[index1]
    data = np.delete(data,index1,axis=0)
    index2 = random.sample(range(num2),num2)
    valid = data[index2]
    test = np.delete(data,index2,axis=0)
    return train,valid,test


def load_data(num = 100,length = 1024,hp = [0,1,2,3],fault_diameter = [0.007,0.014,0.021],split_rate = [0.7,0.1,0.2]):
    bath_path1 = '/root/CRWU_data/Normal Baseline Data/'
    bath_path2 = '/root/CRWU_data/Bearing Fault Data/'
    data_list = []
    file_list = np.array([[105,118,130,106,119,131,107,120,132,108,121,133],  #0.007
                         [169,185,197,170,186,198,171,187,199,172,188,200],  #0.014
                         [209,222,234,210,223,235,211,224,236,212,225,237]])  #0.021
    label = 0
    for i in hp:
        normal_data = open_data(bath_path1,97+i)
        data = deal_data(normal_data,length,label = label)
        data_list.append(data)
    for i in fault_diameter:
        for j in hp:
            inner_num = file_list[int(i/0.007-1),3*j]
            ball_num = file_list[int(i/0.007-1),3*j+1]
            outer_num = file_list[int(i/0.007-1),3*j+2]
            

            inner_data = open_data(bath_path2,inner_num)
            inner_data = deal_data(inner_data,length,label + 1)
            data_list.append(inner_data)

            ball_data = open_data(bath_path2,ball_num)
            ball_data = deal_data(ball_data,length,label + 4)
            data_list.append(ball_data)

            outer_data = open_data(bath_path2,outer_num)
            outer_data = deal_data(outer_data,length,label + 7)
            data_list.append(outer_data)

        label = label + 1

    num_list = []
    for i in data_list:
        num_list.append(len(i))
    min_num = min(num_list)

    if num > min_num:
        print("The number of each class overflow, the maximum number is：%d" %min_num)

    min_num = min(num,min_num)
    train = []
    valid = []
    test = []
    for data in data_list:
        data = data[0:min_num,:]
        a,b,c = split_data(data,split_rate)
        train.append(a)
        valid.append(b)
        test.append(c)

    train = np.reshape(train,(-1,length+1))
    train = train[random.sample(range(len(train)),len(train))]
    train_data = train[:,0:length]
    train_label = train[:,length]
    onehot_encoder = preprocessing.OneHotEncoder(sparse_output=False)
    train_label = train_label.reshape(len(train_label), 1)
    train_label = onehot_encoder.fit_transform(train_label)

    valid = np.reshape(valid,(-1,length+1))
    valid = valid[random.sample(range(len(valid)),len(valid))]
    valid_data = valid[:,0:length]
    valid_label = valid[:,length]
    valid_label = valid_label.reshape(len(valid_label), 1)
    valid_label = onehot_encoder.fit_transform(valid_label)

    test = np.reshape(test,(-1,length+1))
    test = test[random.sample(range(len(test)),len(test))]
    test_data = test[:,0:length]
    test_label = test[:,length]
    test_label = test_label.reshape(len(test_label), 1)
    test_label = onehot_encoder.fit_transform(test_label)


    return train_data,train_label,valid_data,valid_label,test_data,test_label


if __name__ == "__main__":
    train_dataset,train_label,_,_,test_dataset,test_label = load_data()