import numpy as np
from scipy import stats

def ConvertDatasetToBinary(dataset):
    N = len(dataset)
    if(N<=0):
        return None
    
    DIM = len(dataset[0])
    
    height_bt = 3
    type_root = "mean"
    
    binset = np.empty([N,1])
    
    for i_dim in range(0,DIM):
        data_dim = dataset[:,i_dim]
        bin_dim = ConvertContinuousToBinary(data_dim, height_bt, type_root)
        
        binset = np.column_stack((binset,bin_dim))
    
    binset = binset[:,1:]
    print(binset)
    print(binset.shape)
    
    return binset

def ConvertContinuousToBinary(array_data, height_bt, type_root):
#     print(array_data)
    N = len(array_data)
    mean_data = np.mean(array_data)
    median_data = np.median(array_data)
    min_data = np.amin(array_data)
    max_data = np.amax(array_data)
        
    ## Temporarily only accept non-negative values
    if(min_data < 0):
        return None
    
    value_root = mean_data
    if(type_root == "mean"):
        value_root = mean_data
    elif(type_root == "median"):
        value_root = median_data
        
    array_bin_data = np.empty((N,height_bt))
    
    for i_n in range(0, N):
        data = array_data[i_n]
        value_check = value_root
        bin_data = np.empty((1,height_bt))
        
        for i_height in range(0, height_bt):
            bin_i = None
            
            if(data <= value_check):
                # Bipolar
#                 bin_i = -1
                bin_i = 0
                value_check = value_check - value_check/2
            else:
                bin_i = 1
                value_check = value_check + value_check/2
                
            bin_data[0,i_height] = bin_i
                    
#         print(array_bin_data)
        array_bin_data[i_n] = bin_data
        

#     print(array_bin_data)
    return array_bin_data


def ConvertContinuousToDiscreteK(array_data, num_bins):
    N = len(array_data)
    
    k_statistic, k_edges, array_K_data = stats.binned_statistic(array_data, array_data, bins = num_bins)

    return array_K_data


class NodeBT:
    def __init__(self, val_m, val_type, height):
        self.val_m = val_m
        self.val_type = val_type
        self.left = None
        self.right = None
        self.delta = None
        self.height = height
        
    def ConvertToBin(self, val):
        
        digit = -1
        digits_next = None
#         print(val)
#         print(self.val_m)
        
        if(val <= self.val_m):
            # Bipolar
#             digit = -1
            digit = 0
        else:
            digit = 1
        
        if(self.height > 1):
            if(val <= self.val_m):
                digits_next = self.left.ConvertToBin(val)
            else:
                digits_next = self.right.ConvertToBin(val)

            digits_rtn = np.insert(digits_next, 0, digit)
            
            return digits_rtn
        else:
            digits_rtn = np.array([digit])
            
            return digits_rtn
        
    def PrintInOrder(self):
        if(self.left is not None):
            self.left.PrintInOrder()
        
        print(self.val_m)
        
        if(self.right is not None):
            self.right.PrintInOrder()

def ConvertDatasetToBinary_V2(data_train, data_test, height_bt, type_root):
    N_train = len(data_train)
    N_test= len(data_test)
    if(N_train<=0 or N_test <= 0):
        return None
    
    DIM = len(data_train[0])
    
#     height_bt = 3
#     type_root = "half"
    
    data_train_B = np.empty([N_train,1])
    data_test_B = np.empty([N_test,1])
    
    for i_dim in range(0,DIM):
        data_train_dim = data_train[:,i_dim]
        data_test_dim = data_test[:,i_dim]
        
        data_train_dim_B, data_test_dim_B = ConvertContinuousToBinary_V2(data_train_dim, data_test_dim, height_bt, type_root)
        
#         binset = np.column_stack((binset,bin_dim))
        data_train_B = np.column_stack((data_train_B,data_train_dim_B))
        data_test_B = np.column_stack((data_test_B,data_test_dim_B))
    
#     binset = binset[:,1:]
    data_train_B = data_train_B[:,1:]
    data_test_B = data_test_B[:,1:]
#     print(data_train)
#     print(data_train_B)
#     print(data_train_B.shape)
#     print(data_test)
#     print(data_test_B)
#     print(data_test_B.shape)
    
    return data_train_B, data_test_B
        
def ConvertContinuousToBinary_V2(data_train, data_test, height_bt, type_root):
        
    root_bt = HelperCreateBT(data_train, height_bt, type_root)
#     print("-----------------------------------------")
#     root_bt.PrintInOrder()
    data_train_bin = HelperConvertToBin(data_train, root_bt)
    data_test_bin = HelperConvertToBin(data_test, root_bt)

    return data_train_bin, data_test_bin

def HelperCreateBT(data, height_bt, type_root):
    
    N = len(data)
    mean_data = np.mean(data)
    median_data = np.median(data)
    min_data = np.amin(data)
    max_data = np.amax(data)
    
    ## Temporarily only accept non-negative values
    if(min_data < 0):
        return None
    
    value_root = mean_data
    if(type_root == "mean"):
        value_root = mean_data
    elif(type_root == "median"):
        value_root = median_data
    elif(type_root == "half"):
        value_root = mean_data
    
    
    ## Temporarily only works for "half" type
    node_root = NodeBT(value_root, type_root, height_bt)
    node_root.delta = value_root/2
    
    queue = []
    queue.append(node_root)
    
    while(len(queue) > 0):
        node = queue.pop(0)
        
        if(node.height > 1):
            node_left = NodeBT(node.val_m - node.delta, type_root, node.height - 1)
            node_left.delta = node.delta/2
            node.left = node_left
            
            queue.append(node_left)
            
            node_right = NodeBT(node.val_m + node.delta, type_root, node.height - 1)
            node_right.delta = node.delta/2
            node.right = node_right
            
            queue.append(node_right)
            
    return node_root
    
def HelperConvertToBin(data, root_bt):
    
    N = len(data)
    height_bt = root_bt.height
    data_bin = np.empty((N, height_bt))
    
    for i_n in range(0, N):
        val = data[i_n]
        digits = root_bt.ConvertToBin(val)
                
        data_bin[i_n] = digits
    
    return data_bin


def ConvertDatasetToDiscreteK_V2(data_train, data_test, num_bins):
    
    N_train = len(data_train)
    N_test= len(data_test)
    if(N_train<=0 or N_test <= 0):
        return None
    
    DIM = len(data_train[0])
        
    data_train_K = np.empty([N_train,1])
    data_test_K = np.empty([N_test,1])
    
    for i_dim in range(0,DIM):
        data_train_dim = data_train[:,i_dim]
        data_test_dim = data_test[:,i_dim]
        
        data_train_dim_K, data_test_dim_K = ConvertContinuousToDiscreteK_V2(data_train_dim, data_test_dim, num_bins)
        
        data_train_K = np.column_stack((data_train_K,data_train_dim_K))
        data_test_K = np.column_stack((data_test_K,data_test_dim_K))
    
    data_train_K = data_train_K[:,1:]
    data_test_K = data_test_K[:,1:]
#     print(data_train)
#     print(data_train_K)
#     print(data_train_K.shape)
    
#     print(data_test)
#     print(data_test_K)
#     print(data_test_K.shape)
    
#     print(sum(abs(data_train_K-data_test_K)))
    
    return data_train_K, data_test_K
    

def ConvertContinuousToDiscreteK_V2(data_train, data_test, num_bins):
    
    N = len(data_train)
    
#     k_statistic, k_edges, data_train_K = stats.binned_statistic_dd(data_train, data_train, bins = num_bins)
#     ret_train = stats.binned_statistic_dd(data_train, data_train, bins = num_bins)
    ret_train = stats.binned_statistic_dd(data_train, data_train, bins = num_bins)
    data_train_K = ret_train.binnumber
    ret_test = stats.binned_statistic_dd(data_test, data_test, binned_statistic_result=ret_train)
    data_test_K = ret_test.binnumber
    
#     print("--------------------------------------------------------------------")
#     print(ret_train.bin_edges)
#     print(data_train_K)
#     print(data_test_K)
#     print(sum(data_train_K-data_test_K))

    return data_train_K, data_test_K