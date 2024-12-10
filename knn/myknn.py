"""
@encoding: utf-8
@File  : knn.py
@Author: HENRY C
@Desc :
@Date  :  2024/12/02
"""
from sklearn.datasets import fetch_openml
import numpy as np
import pandas

class KNN:
    """
    init 中的参数应该只与算法本身相关
    训练数据和测试数据应该由外部给予
    """
    def __init__(self,k,label_nums):
        self.k=k
        self.label_nums=label_nums
    def fit(self,train_data_x,train_data_y):
        self.train_data_x=train_data_x
        self.train_data_y=train_data_y
    def get_min_indices(self,x):
        dis_list = list(map(lambda tmp: np.sqrt(np.sum(np.square(tmp-x))), self.train_data_x))

        return np.argsort(dis_list)[0:self.k]
    def get_label(self,x):
        min_dis_indices=self.get_min_indices(x)
        count_list=np.zeros(self.label_nums)
        for index in min_dis_indices:
            count_list[self.train_data_y[index]]+=1
        return np.argmax(count_list)
    def predict(self,test_data_x):
        predict_list=np.zeros(len(test_data_x))
        for i ,x in enumerate(test_data_x):
            predict_list[i]=self.get_label(x)
        return predict_list
    def get_correct_rate(self,test_data_x,test_data_y):
        predict_list=self.predict(test_data_x)
        right_count=0
        for i,label in enumerate(test_data_y):
            if predict_list[i]==label:
                right_count+=1
        print(f"K={self.k} correct_rate:{right_count/len(test_data_y)*100:.1f}%")

if __name__ == "__main__":


    # 加载 MNIST 数据集
    mnist = fetch_openml('mnist_784', version=1)

    # 获取数据和标签
    x = mnist.data
    y = mnist.target
    x=x.to_numpy(dtype=int)
    y=y.to_numpy(dtype=int)

    # 切片前 6000 个样本
    x_train = x[:6000,:]
    y_train = y[:6000]

    x_test=x[6000:,:]
    y_test=y[6000:]

    for k in range(1,10):
        knn = KNN(k, label_nums=10)
        knn.fit(x_train,y_train)
        knn.get_correct_rate(x_test,y_test)
