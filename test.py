import numpy as np
import pandas as pd
import warnings
 
 
# 这里的意思是求权重之前我们需要进行一致性检验
# 参考：https://zhuanlan.zhihu.com/p/101505929
# 参考：https://blog.csdn.net/knighthood2001/article/details/127519604?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-2-127519604-blog-98480769.pc_relevant_recovery_v2&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-2-127519604-blog-98480769.pc_relevant_recovery_v2&utm_relevant_index=3
# 一致性检验
def calculate_weight(data):
    RI = (0, 0, 0.52, 0.89, 1.12, 1.26, 1.36, 1.41, 1.46, 1.49, 1.52, 1.54, 1.56, 1.58, 1.59)
    # 转化为array类型的对象
    in_matrix = np.array(data)
    n, n2 = in_matrix.shape
    # 判断矩阵是否为方阵，而且矩阵的大小为n，n2
    if n != n2:
        print("不是一个方阵，所以不能进行接下来的步骤")
        return None
    for i in range(0, n):
        for j in range(0, n2):
            if np.abs(in_matrix[i, j] * in_matrix[j, i] - 1) > 1e-7:
                raise ValueError("不为正互反矩阵")
    eig_values, eig_vectors = np.linalg.eig(in_matrix)
    # eigvalues为特征向量，eigvectors为特征值构成的对角矩阵（而且其他位置都为0，对角元素为特征值）
    max_index = np.argmax(eig_values)
    # argmax为获取最大特征值的下标,而且这里是获取实部
    max_eig = eig_values[max_index].real
    # 这里max_eig是最大的特征值
    eig_ = eig_vectors[:, max_index].real
    eig_ = eig_ / eig_.sum()
    if n > 15:
        CR = None
        warnings.warn(("无法判断一致性"))
    else:
        CI = (max_eig - n) / (n - 1)
        if RI[n - 1] != 0:
            CR = CI / RI[n - 1]
        if CR < 0.1:
            print("矩阵的一致性可以被接受")
        else:
            print("矩阵的一致性不能被接受")
    return max_eig, CR, eig_
 
 
# 特征值法求权重
def calculate_feature_weight(matrix, n):
    # 特征值法主要是通过求出矩阵的最大特征值和对应的特征向量，然后对其特征向量进行归一化，最后获得权重
    eigValue, eigVectors = np.linalg.eig(matrix)
    max_index = np.argmax(eigValue)
    max_eig = eigValue[max_index].real
    eig_ = eigVectors[:, max_index].real
    # 返回的是特征向量，而且max_index为最大的特征值，在这里一般为n
    eig_ = eig_ / eig_.sum()
    # 这里返回的是特征向量
    return eig_
 
 
# 算术平均法求权重
def calculate_arithemtic_mean(matrix):
    n = len(matrix)
    matrix_sum = sum(matrix)
    normalA = matrix / matrix_sum  # 归一化处理
    average_weight = []
    for i in range(0, n):
        # 按照列求和
        temSum = sum(normalA[i])
        average_weight.append(temSum / n)
    return np.array(average_weight)
 
 
# 几何平均法求权重
def calculate_metric_mean(metrix):
    n = len(metrix)
    # 1表示按照行相乘，得到一个新的列向量,每行相乘获得一个列向量，所以用prod函数，
    vector = np.prod(metrix, 1)
    tem = pow(vector, 1 / n)
    # 开n次方
    # 归一化处理
    average_weight = tem / sum(tem)
    return average_weight
 
 
if __name__ == "__main__":
    # 准则重要性矩阵,即为我们在这里的时候要输入准则性矩阵
    criteria = np.array([[1, 2, 7, 5, 5],
                         [1 / 2, 1, 4, 3, 3],
                         [1 / 7, 1 / 4, 1, 1 / 2, 1 / 3],
                         [1 / 5, 1 / 3, 2, 1, 1],
                         [1 / 5, 1 / 3, 3, 1, 1]])
 
    # 对每个准则，方案优劣排序,即为方案层也会有一个一致矩阵，所以需要判断
    b1 = np.array([[1, 1 / 3, 1 / 8], [3, 1, 1 / 3], [8, 3, 1]])
    b2 = np.array([[1, 2, 5], [1 / 2, 1, 2], [1 / 5, 1 / 2, 1]])
    b3 = np.array([[1, 1, 3], [1, 1, 3], [1 / 3, 1 / 3, 1]])
    b4 = np.array([[1, 3, 4], [1 / 3, 1, 1], [1 / 4, 1, 1]])
    b5 = np.array([[1, 4, 1 / 2], [1 / 4, 1, 1 / 4], [2, 4, 1]])
    b = [b1, b2, b3, b4, b5]
    matrix_in = criteria
    max_eigen, CR, criteria_eigen = calculate_weight(matrix_in)
    print("准则层：最大特征值：{:.5f},CR={:<.5f},检验{}通过".format(max_eigen, CR, '' if CR < 0.1 else "不"))
    print("准则层权重为{}\n".format(criteria_eigen))
    max_eigen_list = []
    CR_list = []
    eigen_list = []
    for i in b:
        max_eigen, CR, eigen = calculate_weight(i)
        max_eigen_list.append(max_eigen)
        CR_list.append(CR)
        eigen_list.append(eigen)
    pd_print = pd.DataFrame(eigen_list, index=["准则" + str(i) for i in range(0, criteria.shape[0])],
                           columns=["方案" + str(i) for i in range(0, b[0].shape[0])])
    pd_print.loc[:, '最大特征值'] = max_eigen_list
    pd_print.loc[:, 'CR'] = CR_list
    pd_print.loc[:, '一致性检验'] = pd_print.loc[:, 'CR'] < 0.1
    print("方案层")
    print(pd_print)
    # 目标层
    # np.dot()函数为向量点积和矩阵乘法，即为AHP最后的目的是将准则层的的特征向量和方案层的特征向量进行矩阵乘法，而且最后是1*方案层的矩阵，
    # criteria_eigen的shape为（1,5），而且eight_list为（5,3）的矩阵
    # 而且reshape类似于转置矩阵的作用，所以使得原来为（5，）变成（1，5）
    object = np.dot(criteria_eigen.reshape(1, -1), np.array(eigen_list))
    print("\n目标层", object)
    print("最优选择方案{}".format(np.argmax(object)))