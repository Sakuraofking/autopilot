# 建立，训练保存自动驾驶模型
# 1.导入第三方库
import cv2
import keras.callbacks
import numpy as np
import pandas as pd#读写csv
from keras.models import Sequential  # 序贯模型
from sklearn.model_selection import train_test_split  # 机器学习工具集
from keras.optimizers import adam_v2  # 优化函数
from keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda  # 卷积层，全连接层，池化层，变平层，数据运算
from keras.callbacks import ModelCheckpoint  # 运行训练获取反馈
from data_preprocessing import image_preprocessing, image_normalized  # 导入图像处理函数
from data_preprocessing import image_height, image_width, image_channels  # 导入图像处理的图像信息

# 2.初始化变量
Input_Size = (image_height, image_width, image_channels)  # 输入图像大小
data_path = './data/'
test_ration = 0.1  # 测试比例
batch_size = 200  # 100个图片算一组
batch_num = 400  # 200一个巡回自我修正，实际训练量100x200
epoch = 100  # 50个巡回建立模型，实际训练量100x200x50


# 3.导入数据
def load_data(data_path):
    pd_read_csv = pd.read_csv(data_path + 'driving_log.csv',
                              names=['center', 'left', 'right', 'steering_angle', '_', '__', '___'])#读取csv文件，并且赋值取名
    # print(pd_read_csv)
    X = pd_read_csv[['center', 'left', 'right']].values#获取对应的三幅图像
    Y = pd_read_csv['steering_angle'].values#获取对应转向角
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_ration, random_state=0)  # 划分测试集和训练集，测试集10%，训练集90%，方法为零代码
    return X_train, X_test, Y_train, Y_test


# load_data(data_path)
# 4.批量生产数据
def data_generator(data_path, batch_size, X_data, Y_data, tarin_flag):
    X_container = np.empty([batch_size, image_height, image_width, image_channels])  # 创建图像容器，容器大小，长，宽，通道数
    Y_container = np.empty(batch_size)  # 创建方向角容器
    #print(X_container.shape)
    while True:
        ii=0
        for index in np.random.permutation(X_data.shape[0]):#从训练集随机抽取一个X数据的编号
            center,left,right=data_path+X_data[index]#拆分为三幅图像
            #print(center,left,right)
            steering_angle=Y_data[index]#把转向角也抽出来
            #print(steering_angle)
            if tarin_flag and np.random.rand()<0.4:#取60%的训练数据进行图像处理
                image,steering_angle=image_preprocessing(center, left, right, steering_angle)#图像预处理
            else:#剩余的40%作为测试
                image=cv2.imread(center)
            X_container[ii]=image_normalized(image)
            Y_container[ii]=steering_angle#图像归一化之后放到容器里，一个图像，一个方向角
            ii += 1
            if ii == batch_size:#容量达到容器上限后跳出循环
                break
            #print(Y_container)
        yield X_container,Y_container  # yield可以在下次运行时从当前程序执行

#5.搭建CNN
def build_model():
    model=Sequential()#序贯结构
    model.add(Lambda(lambda x:x/127.5-1,input_shape=Input_Size))#将数据处理为-1到1的单精度数值
    model.add(Conv2D(
        filters=24,#卷积核的数量
        kernel_size=(5,5),#卷积核大小
        strides=(2,2),#步长
        activation='elu'#激活函数elu，优化操作，防止梯度爆炸
    ))
    model.add(Conv2D(
        filters=36,  # 卷积核的数量
        kernel_size=(5, 5),  # 卷积核大小
        strides=(2, 2),
        activation='elu'  # 激活函数elu
    ))
    model.add(Conv2D(
        filters=48,  # 卷积核的数量
        kernel_size=(5, 5),  # 卷积核大小
        strides=(2, 2),
        activation='elu'  # 激活函数elu
    ))
    model.add(Conv2D(
        filters=64,  # 卷积核的数量
        kernel_size=(3, 3),  # 卷积核大小
        strides=(1, 1),
        activation='elu'  # 激活函数elu
    ))
    model.add(Conv2D(64,(3,3),strides=(1,1),activation='elu'))
    #model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='elu'))
    model.add(Dropout(0.5))#数据过多需要丢弃一些，池化为丢一些特征过于明显的元素，这里是随机丢一半
    model.add(Flatten())#扁平层，图像从dropout拉直变成一维图像
    model.add(Dense(100,activation='elu'))
    model.add(Dense(50,activation='elu'))
    model.add(Dense(10,activation='elu'))
    model.add(Dense(1))#全连接层
    model.summary()#总结提醒
    return model

#6.训练模型
X_train,X_test,Y_train,Y_test=load_data(data_path)#导入训练模型并进行分类
model=build_model()#建立卷积模型
#优化器Adam，sgd，adagrad，ada
model.compile(optimizer=adam_v2.Adam(learning_rate=0.0001),loss='mse')#配置训练方法规定训练优化器，损失函数和评测标准，lr学习率，decay损失率
checkpoint=ModelCheckpoint('model_test{epoch:02d}.h5',
                           monitor='val_loss',#需要监测的值，loss（训练集损失值）,val_loss（测试集损失值），acc、val_acc
                           verbose=1,#信息展示模式
                           save_best_only=1,#保存最佳值
                           mode='auto'#自动协商监测值哪一个好
                           )#回调函数
'''
训练数据时，fit（）把所有数据一次性扔进内存，适用于数量小且简单的数据集
fit_generator()动态分批次训练数据，而且在训练过程中可以增强数据
'''
model.fit_generator(data_generator(data_path,batch_size,X_train,Y_train,1),
                    steps_per_epoch=batch_num,#一轮的数据量
                    epochs=epoch,#进行多少轮训练
                    max_queue_size=5,#后备队列列数
                    validation_data=data_generator(data_path,batch_size,X_test,Y_test,0),#测试集喂料机
                    validation_steps=1,#测试集进行的次数
                    callbacks=[checkpoint]#回调函数
                    )
model.save('model1.h5')


#X_train,X_test,Y_train,Y_test=load_data(data_path)
#data_generator(data_path,batch_size,X_train,Y_train,1)
