# 1.扩充数据量 2.丰富数据类型 3.数据归一化
# 1.导入第三方库
import cv2
import numpy as np#数据处理

# 2.初始化变量
image_height, image_width, image_channels = 66, 200, 3#图片的高，宽，通道数
center, left, right = './test/center.jpg', './test/left.jpg', './test/right.jpg'#图片地址
steering_angle = 0.0  # 代表送给模型的方向角变量


# 3.选择图像
def image_choose(center, left, right, steering_angle):
    choice = np.random.choice(3)  # 0,1,2随机摇一个数
    if choice == 0:
        image_name = left  # 确定随机图像的名字
        steering_angle_1 = 0.2  # 调整方向角
    if choice == 1:
        image_name = center
        steering_angle_1 = 0.0  # 修正值，一个经验值
    if choice == 2:
        image_name = right
        steering_angle_1 = -0.2
    image = cv2.imread(image_name)#读取图像
    steering_angle = steering_angle + steering_angle_1  # 加入方向偏移量之后返回给方向角
    #cv2.imshow('image_choose', image)
    #cv2.waitKey(0)
    return image, steering_angle  # 返回读取到的图像和修正后的方向角


# image_choose(center,left,right, steering_angle)
# 4.图像的变换
def image_translate(image, steering_angle):
    range_x, range_y = 100, 10  # 图像移动范围
    tran_x = range_x * (np.random.rand() - 0.5)#随机平移x
    tran_y = range_y * (np.random.rand() - 0.5)#随机平移y
    steering_angle = steering_angle + tran_x * 0.002  # 角度随图像平移的最佳数值
    tran_m = np.float32([[1, 0, tran_x], [0, 1, tran_y]])  # 图像平移公式
    image = cv2.warpAffine(image, tran_m, (image.shape[1], image.shape[0]))  # 仿射变换，1为宽，0为高
    # cv2.imshow('image_translate',image)
    # cv2.waitKey(0)
    return image, steering_angle


# image,steering_angle=image_choose(center,left, right, steering_angle)#通过图像选择获取图像和方位角
# image_translate(image,steering_angle)#对获取到的图像进行平移变换
# 5.归一化图像（统一图像数据）
def image_normalized(image):
    image = image[60:-25, :, :]  # 去掉机箱盖和天空，获取这个范围内的图像，保留宽度和通道数
    image = cv2.resize(image, (image_width, image_height), cv2.INTER_AREA)  # 统一图像的大小，使用inter_area裁剪
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)  # 计算机需要的是YUV
    # cv2.imshow('image_normalized',image)
    # cv2.waitKey(0)
    return image


# image_normalized(image)
# 6.预处理图像综合
def image_preprocessing(center, left, right, steering_angle):
    image, steering_angle = image_choose(center, left, right, steering_angle)  # 通过图像选择获取图像和方向角
    image, steering_angle = image_translate(image, steering_angle)  # 对获取到的图像进行平移变换
    return image, steering_angle  # 返回归一化前处理的图像和方向角


#7. 作为主函数运行
if __name__ == '__main__':  # 如果这句话是单独运行就运行下面的语句，如果作为主函数则调用这一段
    image, steering_angle = image_preprocessing(center, left, right, steering_angle)
    image = image_normalized(image)
    print(steering_angle)
    cv2.imshow('image_normalized', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
