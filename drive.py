# 驾驶车辆，验证训练模型
# 1.导入第三方库
import socketio#web通信架构
from flask import Flask#web微型架构框架
import eventlet.wsgi#web server gateway interface web服务器网关接口，用于连接服务端与客户端
import cv2
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from keras.models import load_model
from data_preprocessing import image_normalized#图像处理函数
#读取无人驾驶训练模型
model=load_model('model2.2.h5')
# 2.初始化变量
max_speed = 20#最大速度
steering_angle = -0.02#转向角
throttle = 0.3#油门，最大为1

# 3.创建网络连接
sio = socketio.Server()  # 创建服务器
app = Flask(__name__)  # 定义简单的服务器应用
app = socketio.WSGIApp(sio, app)  # 连接sio和app


# 4.发送控制参数
def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
        }
    )


# 5.控制汽车运行
@sio.on('connect')  # 连接成功时需要的操作
def on_connect(sid, environ):  # 随机端口号与交流信息
    print('连接成功')
@sio.on('telemetry')  # 收到数据的操作
def on_telemetry(sid, data):
    if data:
        #print('我收到数据了')
        #print(data)
        speed = float(data['speed'])  # 读到的数据转换成浮点数
        print('Speed=',speed)
        '''
        把模型中小车前方的图像从缓存读取到内存，并且打开图像
        bytesio把图像读取到内存，二进制数据
        stringio，图像读取到内存，string数据（字符串）
        b64decode数据编码转换为二进制
        '''
        image=Image.open(BytesIO(base64.b64decode(data['image'])))
        image=np.array(image)#列表转数组
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)#BGR转化为RGB
        cv2.imshow('Image from Udacity',image)#显示进行中的图像
        cv2.waitKey(1)#图像归一化
        image=image_normalized(image)
        steering_angle=float(model.predict(np.array([image])))

        class SimplePIController:  # 定速巡航
            def __init__(self, Kp, Ki):
                self.Kp = Kp
                self.Ki = Ki
                self.set_point = 0.
                self.error = 0.
                self.integral = 0.

            def set_desired(self, desired):
                self.set_point = desired

            def update(self, measurement):
                # proportional error
                self.error = self.set_point - measurement

                # integral error
                self.integral += self.error
                # print(self.integral)
                return self.Kp * self.error + self.Ki * self.integral

        controller = SimplePIController(0.1, 0.002)
        set_speed = 18
        controller.set_desired(set_speed)

        throttle = controller.update(speed)
        #throttle = 1.0 - steering_angle ** 2 - (speed / max_speed) ** 2  # 定速巡航
        send_control(steering_angle, throttle)  # 送出控制信号



@sio.on('disconnect')  # 当断开连接后
def on_disconnect(sid):
    print('断开连接！')


eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
#求一份工作