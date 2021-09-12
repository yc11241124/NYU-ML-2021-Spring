import os
import numpy as np

def LoadData_Abalone():
    
    file_original = open("abalone.data", "r")
    Lines = file_original.readlines()

    X = []
    Y = []
    
    for line in Lines:
        data = []
        line = line.replace('\n','')
        line_split = line.split(',')

        label_int = int(line_split[len(line_split)-1])
        Y.append(label_int)

    #     if label_int >= 1 and label_int <= 9:
    #         list_data.append(str(1))
    #     else:
    #         list_data.append(str(0))

        if(line_split[0]=="F"):
            data.append(1)
        elif(line_split[0]=="M"):
            data.append(2)
        elif(line_split[0]=="I"):
            data.append(3)

        for i in range(1,len(line_split)-1):
            val = float(line_split[i])

            data.append(val)

        X.append(data)

    file_original.close()
    
    X = np.array(X)
    Y = np.array(Y)
    
    return X, Y

def LoadData_Banknote():
    
    file_original = open("banknote.data", "r")
    Lines = file_original.readlines()

    X_raw = []
    Y = []
    
    for line in Lines:
        data = []
        line = line.replace('\n','')
        line_split = line.split(',')

        label_int = int(line_split[len(line_split)-1])
        Y.append(label_int)

        for i in range(1,len(line_split)-1):
            val = float(line_split[i])

            data.append(val)

        X_raw.append(data)

    file_original.close()
    
    X_raw = np.array(X_raw)
    
    X = np.zeros(X_raw.shape)
    for i in range(0,X_raw.shape[1]):
        X_i = X_raw[:,i]
        min_i = np.amin(X_i)
#         print(np.amin(X_i))
        min_i = 100 #np.amin(X_i)
        X_shifted_i = X_i + np.absolute(min_i)
        X[:,i] = X_shifted_i
        
    Y = np.array(Y)
    
    return X, Y

def LoadData_Turnover():
    file_original = open("turnover.data", "r")
    Lines = file_original.readlines()

    X = []
    Y = []

    for line in Lines:
        data = []
        line = line.replace('\n','')
        line_split = line.split(',')

        label_int = int(line_split[len(line_split)-1])
        Y.append(label_int)

        for i in range(1,len(line_split)-1):
            val = float(line_split[i])

            data.append(val)

        X.append(data)

    file_original.close()

    X = np.array(X)
    Y = np.array(Y)

    return X, Y
