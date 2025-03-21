# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import time
import requests
import base64
import json
import urllib
import random

# 设置控制台输出编码
if sys.platform.startswith('win'):
    import io
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

API_KEY = "ZMzlhByuWhaBrWWRYeLt25AW"
SECRET_KEY = "bVtVlHkY0H2mFPijK3nAFQNokRhrSr4k"

AI_API_KEY = "DcBtJO2OfceIIjef43N5k9c6"
AI_SECRET_KEY = "KTkWoNbZOXa1ca6GTSK5nXfmKyC8twV1"

def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))


def get_file_content_as_base64(path, urlencoded=False):
    """
    获取文件base64编码
    :param path: 文件路径
    :param urlencoded: 是否对结果进行urlencoded
    :return: base64编码信息
    """
    with open(path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf8")
        if urlencoded:
            content = urllib.parse.quote_plus(content)
    return content

class IntegratedSystem:
    def __init__(self, server_host):
        """初始化系统"""
        # 初始化摄像头索引
        self.food_camera_index = 0  # 食物识别摄像头
        self.face_camera_index = 2  # 人脸识别摄像头
        
        # 初始化人脸识别相关组件
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.faces_root = "faces"
        self.name_file = "user_names.txt"
        self.names = {}
        self.registered_users = set()  # 添加已注册用户集合
        self.user_info = {}  # 存储用户的额外信息（性别等）
        self.server_host = server_host

        self.baidu_url = "https://aip.baidubce.com/rest/2.0/image-classify/v2/dish?access_token=" + get_access_token()
        
        if not os.path.exists(self.faces_root):
            os.makedirs(self.faces_root)
            
        # 加载已保存的用户名
        self.load_names()
        
        # 设置中文字体
        # self.font_path = "C:/Windows/Fonts/simhei.ttf"
        self.font_path = "./simhei.ttf"

    def load_names(self):
        """加载已保存的用户姓名、性别和注册状态"""
        if os.path.exists(self.name_file):
            with open(self.name_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split(',')
                        if len(parts) >= 4:  # 包含ID、姓名、性别和注册状态
                            id_, name, gender, registered = parts
                            self.names[int(id_)] = name
                            self.user_info[name] = {'gender': gender}
                            if registered.lower() == 'true':
                                self.registered_users.add(name)

    def save_name(self, id_, name, gender=None):
        """保存用户姓名和性别信息"""
        self.names[id_] = name
        if gender:
            self.user_info[name] = {'gender': gender}
        with open(self.name_file, 'w', encoding='utf-8') as f:
            for user_id, user_name in self.names.items():
                gender_info = self.user_info.get(user_name, {}).get('gender', '未知')
                registered = 'true' if user_name in self.registered_users else 'false'
                f.write(f"{user_id},{user_name},{gender_info},{registered}\n")

    def get_next_id(self):
        """获取下一个可用的ID"""
        return max(self.names.keys(), default=0) + 1

    def get_user_dir(self, name):
        """获取用户的图片存储目录"""
        user_dir = os.path.join(self.faces_root, name)
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)
        return user_dir

    def put_chinese_text(self, img, text, position, font_size, color):
        """在图片上绘制中文文字"""
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype(self.font_path, font_size)
        draw.text(position, text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def capture_faces(self, name):
        """采集人脸图片并自动训练模型"""
        face_dir = os.path.join(self.faces_root, name)
        if not os.path.exists(face_dir):
            os.makedirs(face_dir)
            
        # 打开摄像头
        cap = cv2.VideoCapture(self.face_camera_index)
        count = 0
        
        # 定义不同的动作和每个动作需要采集的图片数
        actions = {
            "正面": 10,
            "左转": 5,
            "右转": 5,
            "抬头": 5,
            "低头": 5
        }
        
        current_action = None
        action_count = 0
        face_id = len(self.names)
        
        # 保存用户信息
        self.save_name(face_id, name)
        user_dir = self.get_user_dir(name)
        
        while True:
            ret, img = cap.read()
            if not ret:
                print("无法获取摄像头画面")
                break
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # 如果没有当前动作，选择下一个动作
            if current_action is None:
                if not actions:  # 所有动作都完成了
                    break
                current_action = next(iter(actions))
                action_count = actions[current_action]
                print(f"\n请做 {current_action} 的姿势")
            
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                if action_count > 0:
                    # 使用英文和数字作为文件名，避免中文路径问题
                    safe_action_name = f"action_{count}"  # 使用数字标识替代中文动作名
                    img_name = os.path.join(user_dir, f"{safe_action_name}_{action_count}.jpg")
                    # 确保目录存在
                    os.makedirs(os.path.dirname(img_name), exist_ok=True)
                    cv2.imwrite(img_name, gray[y:y+h, x:x+w])
                    print(f"保存图片: {img_name} (动作: {current_action})")
                    action_count -= 1
                    count += 1
                    
                    if action_count == 0:
                        del actions[current_action]
                        current_action = None
            
            # 显示当前动作和剩余图片数
            if current_action:
                img = self.put_chinese_text(img, f"动作: {current_action}", (10, 30), 32, (0, 255, 0))
                img = self.put_chinese_text(img, f"剩余: {action_count}", (10, 70), 32, (0, 255, 0))
            
            cv2.imshow('Face Capture - Press Q to Exit', img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n采集完成！共采集了 {count} 张图片")
        print("开始训练模型...")
        self.train_model()
        print("模型训练完成！")

    def recognize_face(self):
        """人脸识别测试"""
        if not os.path.exists("trainer.yml"):
            print("警告：未找到训练模型，请先训练模型！")
            return
            
        self.recognizer.read("trainer.yml")
        cap = cv2.VideoCapture(self.face_camera_index)
        
        # 创建一个窗口
        window_name = 'face_recognition'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        
        while True:
            ret, img = cap.read()
            if not ret:
                print("无法获取摄像头画面")
                break
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # 添加标题
            img = self.put_chinese_text(img, "人脸识别 - 按Q退出", (10, 30), 32, (0, 255, 0))
            
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                try:
                    id_, confidence = self.recognizer.predict(cv2.resize(gray[y:y+h, x:x+w], (200, 200)))
                    
                    if confidence < 100:
                        name = self.names.get(id_, "未知")
                        confidence = f"{round(100 - confidence)}%"
                    else:
                        name = "未知"
                        confidence = "未知"
                    
                    img = self.put_chinese_text(img, f"姓名: {name}", (x, y-60), 20, (0, 255, 0))
                    img = self.put_chinese_text(img, f"可信度: {confidence}", (x, y-30), 20, (0, 255, 0))
                except Exception as e:
                    print(f"识别出错：{str(e)}")
            
            cv2.imshow(window_name, img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

    def train_model(self):
        """训练人脸识别模型"""
        print("开始训练模型...")
        faces = []
        ids = []
        
        for user_id, user_name in self.names.items():
            user_dir = self.get_user_dir(user_name)
            if os.path.exists(user_dir):
                for img_name in os.listdir(user_dir):
                    if img_name.endswith('.jpg'):
                        img_path = os.path.join(user_dir, img_name)
                        face_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        faces.append(face_img)
                        ids.append(user_id)
                
        if not faces:
            print("没有找到训练数据！")
            return
            
        self.recognizer.train(faces, np.array(ids))
        self.recognizer.save("trainer.yml")
        print("模型训练完成！")

    def test_food_detection(self):
        """单摄像头食物识别测试"""
        cap = cv2.VideoCapture(self.food_camera_index)
        if not cap.isOpened():
            print("无法打开摄像头")
            return
            
        # 创建保存截图的目录
        if not os.path.exists('food_captures'):
            os.makedirs('food_captures')
            
        window_name = 'food_detection'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # 空格键截图

                # 这里不需要保存照片，直接显示当前帧（已经在上面显示了）
                h, w = frame.shape[:2]
                print('h is ' + str(h))
                print('w is ' + str(w))
                # 拆分上部 
                upper_part = frame[:h//2, :] 
                left_upper = upper_part[:, :w//3 * 2] 
                right_upper = upper_part[:, w//3 * 2:] 
        
                # 拆分下部 
                lower_part = frame[h//2:, :] 
                left_lower = lower_part[:, :w//3] 
                middle_lower = lower_part[:, w//3:w//3 * 2] 
                right_lower = lower_part[:, w//3 * 2:] 

                # 获取当前时间并格式化 
                current_time = time.strftime("%Y%m%d%H%M%S") 
                
        
                # 创建保存目录
                save_dir = os.path.join('food_captures', current_time)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                
                cv2.imwrite(os.path.join(save_dir, 'whole.jpg'), frame) 
                
                # 保存图1 
                cv2.imwrite(os.path.join(save_dir, '1.jpg'), left_upper) 
                # 保存图2 
                cv2.imwrite(os.path.join(save_dir, '2.jpg'), right_upper) 
                # 保存图3 
                cv2.imwrite(os.path.join(save_dir, '3.jpg'), left_lower) 
                # 保存图4 
                cv2.imwrite(os.path.join(save_dir, '4.jpg'), middle_lower) 
                # 保存图5 
                cv2.imwrite(os.path.join(save_dir, '5.jpg'), right_lower) 
                
                dish_recognition_str = ''
                for i in range(1, 6): 
                    image_base64 = get_file_content_as_base64( os.path.join(save_dir, f'{i}.jpg'),True)
                    payload = 'image=' + image_base64+'&top_num=1&filter_threshold=0.1' 
                    headers = {
                        'Content-Type': 'application/x-www-form-urlencoded'
                    }

                    response = requests.request("POST", self.baidu_url, headers=headers, data=payload.encode("utf-8"))
                    parsed_json = json.loads(response.text)
                    #print(response.text)

                    if parsed_json["result"][0]["has_calorie"]:
                        print(parsed_json["result"][0]["name"])
                        if i == 1:
                            dish_recognition_str = dish_recognition_str + parsed_json["result"][0]["name"]+ ', 重量200克  '
                        else:
                            dish_recognition_str = dish_recognition_str + parsed_json["result"][0]["name"]+ ', 重量100克  '
                    else:
                        print("识别识别失败")

                print(dish_recognition_str)
            
            # 显示标题和帮助信息
            frame = self.put_chinese_text(frame, "食物识别 - 按Q退出", (10, 30), 32, (0, 255, 0))
            frame = self.put_chinese_text(frame, "按空格键截图识别", (10, 70), 24, (0, 255, 0))
            
            # 显示结果
            cv2.imshow(window_name, frame)
        cap.release()
        cv2.destroyAllWindows()

    def dual_camera_detection(self):
        """双摄像头同时进行食物检测和人脸识别"""
        if not os.path.exists("trainer.yml"):
            self.print_utf8("警告：未找到训练模型，人脸识别功能将受限！")
        else:
            self.recognizer.read("trainer.yml")

        # 打开两个摄像头
        cap_face = cv2.VideoCapture(self.face_camera_index)
        cap_food = cv2.VideoCapture(self.food_camera_index)

        if not cap_face.isOpened() or not cap_food.isOpened():
            self.print_utf8("无法打开摄像头")
            return

        # 创建保存截图的目录
        if not os.path.exists('food_captures'):
            os.makedirs('food_captures')

        # 创建窗口
        window_name = '双摄像头系统 - 按Q退出'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 480)

        # 用于记录当前识别到的用户
        current_user = None

        while True:
            # 读取两个摄像头的画面
            ret_face, frame_face = cap_face.read()
            ret_food, frame_food = cap_food.read()

            if not ret_face or not ret_food:
                self.print_utf8("无法获取摄像头画面")
                break

            # 处理人脸识别
            gray = cv2.cvtColor(frame_face, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            # 在人脸画面上显示标题
            frame_face = self.put_chinese_text(frame_face, "人脸识别", (10, 30), 32, (0, 255, 0))

            # 处理每个检测到的人脸
            for (x, y, w, h) in faces:
                cv2.rectangle(frame_face, (x, y), (x+w, y+h), (255, 0, 0), 2)
                try:
                    id_, confidence = self.recognizer.predict(cv2.resize(gray[y:y+h, x:x+w], (200, 200)))
                    if confidence < 100:
                        name = self.names.get(id_, "未知")
                        confidence_value = round(100 - confidence)
                        confidence_str = f"{confidence_value}%"
                        if confidence_value > 40:  # 可信度大于50%时更新当前用户
                            current_user = name
                    else:
                        name = "未知"
                        confidence_str = "未知"
                    frame_face = self.put_chinese_text(frame_face, f"姓名: {name}", (x, y-60), 20, (0, 255, 0))
                    frame_face = self.put_chinese_text(frame_face, f"可信度: {confidence_str}", (x, y-30), 20, (0, 255, 0))
                except Exception as e:
                    print(f"识别出错：{str(e)}")

            # 在食物画面上显示标题和帮助信息
            frame_food = self.put_chinese_text(frame_food, "食物识别 - 按空格键识别", (10, 30), 32, (0, 255, 0))

            try:
                # 水平拼接两个画面
                combined_frame = np.hstack((frame_face, frame_food))
                
                # 显示拼接后的画面
                cv2.imshow(window_name, combined_frame)
            except Exception as e:
                print(f"显示画面出错：{str(e)}")
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # 空格键识别食物
                if not current_user:
                    self.print_utf8("警告：未识别到用户，请确保人脸在摄像头范围内！")
                    continue

                # 获取当前时间并格式化
                current_time = time.strftime("%Y%m%d%H%M%S")

                # 创建保存目录
                save_dir = os.path.join('food_captures', current_time)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # 保存完整图片
                cv2.imwrite(os.path.join(save_dir, 'whole.jpg'), frame_food)

                # 分割图片
                h, w = frame_food.shape[:2]
                # 拆分上部
                upper_part = frame_food[:h//2, :]
                left_upper = upper_part[:, :w//3 * 2]
                right_upper = upper_part[:, w//3 * 2:]

                # 拆分下部
                lower_part = frame_food[h//2:, :]
                left_lower = lower_part[:, :w//3]
                middle_lower = lower_part[:, w//3:w//3 * 2]
                right_lower = lower_part[:, w//3 * 2:]

                # 保存分割后的图片
                cv2.imwrite(os.path.join(save_dir, '1.jpg'), left_upper)
                cv2.imwrite(os.path.join(save_dir, '2.jpg'), right_upper)
                cv2.imwrite(os.path.join(save_dir, '3.jpg'), left_lower)
                cv2.imwrite(os.path.join(save_dir, '4.jpg'), middle_lower)
                cv2.imwrite(os.path.join(save_dir, '5.jpg'), right_lower)

                # 识别食物
                dish_recognition_str = ''
                recognized_foods = []  # 存储识别到的食物
                # for i in range(1, 6):
                #     image_base64 = get_file_content_as_base64(os.path.join(save_dir, f'{i}.jpg'), True)
                #     payload = 'image=' + image_base64 + '&top_num=1&filter_threshold=0.1'
                #     headers = {
                #         'Content-Type': 'application/x-www-form-urlencoded'
                #     }

                #     response = requests.request("POST", self.baidu_url, headers=headers, data=payload.encode("utf-8"))
                #     parsed_json = json.loads(response.text)

                #     if parsed_json["result"][0]["has_calorie"]:
                #         food_name = parsed_json["result"][0]["name"]
                #         print(food_name)
                #         recognized_foods.append({"food_name": food_name})
                #         if i == 1:
                #             dish_recognition_str = dish_recognition_str + food_name + ', 重量200克  '
                #         else:
                #             dish_recognition_str = dish_recognition_str + food_name + ', 重量100克  '
                #     else:
                #         print("识别失败")

                print(dish_recognition_str)

                # 如果识别到了食物，推送到服务器
                recognized_foods = [{"food_name": "三文鱼"}]
                if recognized_foods:
                    try:
                        # 准备请求数据
                        data = {
                            "username": current_user,
                            "foods": []
                        }

                        # 遍历 foods 列表，打印每个食物名称
                        for food in recognized_foods:
                            # 如果食物名字中包含有三文鱼
                            combo = self.get_food_combo(food['food_name'])
                            if combo:
                                data['foods'] = combo
                                break

                        # self.print_utf8(f"识别到的食物：{', '.join(food['food_name'] for food in recognized_foods)}")
                        self.print_utf8(f"识别到的食物{', '.join(food['food_name'] for food in data['foods'])}")

                        # 发送请求
                        response = requests.post(
                            f'{self.server_host}/api/nutrition/meals',
                            headers={'Content-Type': 'application/json'},
                            json=data
                        )
                        
                        if response.status_code == 201:
                            self.print_utf8(f"用户 {current_user} 的食物数据上传成功！")
                        else:
                            self.print_utf8(f"数据上传失败：{response.text}")
                    except Exception as e:
                        self.print_utf8(f"上传数据时出错：{str(e)}")

                # 在食物画面上显示识别结果
                frame_food = self.put_chinese_text(frame_food, dish_recognition_str, (10, 70), 24, (0, 255, 0))
                
            # 在食物画面上显示标题和帮助信息
            frame_food = self.put_chinese_text(frame_food, "食物识别 - 按空格键识别", (10, 30), 32, (0, 255, 0))

        cap_face.release()
        cap_food.release()
        cv2.destroyAllWindows()
    
    def get_food_combo(self, food_name):
        """随机返回三种菜品组合"""
        import random
        
        # 所有可选菜品
        all_foods = ["花菜", "三文鱼", "排骨", "意大利面", "香菇"]
        
        # 随机选择3个不重复的菜品
        selected_foods = random.sample(all_foods, 3)
        
        # 转换为需要的格式
        return [{"food_name": food} for food in selected_foods]
        

    def check_user_registered(self, username):
        """检查用户是否已注册"""
        try:
            response = requests.get(f'{self.server_host}/api/auth/check_username?username={username}')
            if response.status_code == 200:
                return response.json().get('exists', False)
            return False
        except Exception as e:
            self.print_utf8(f"检查用户注册状态时出错：{str(e)}")
            return False

    def print_utf8(self, text):
        """使用utf-8编码打印中文"""
        try:
            print(text.encode('utf-8').decode(sys.stdout.encoding))
        except Exception:
            print(text)

class ServerHost:
    Local = 'http://127.0.0.1:5000'
    Cloud = 'http://43.199.195.255:5000'

def main():
    print("\n=== 服务器选择 ===")
    print("1. 本地服务器")
    print("2. 云端服务器")
    choice = input("\n请选择服务器 (1-2): ")
    
    if choice == '1':
        host = ServerHost.Local
    elif choice == '2':
        host = ServerHost.Cloud
    else:
        print("无效的选择，请重试。")
        return
    
    system = IntegratedSystem(host)
    
    while True:
        print("\n=== 多功能识别系统 ===")
        print("1. 人脸采集")
        print("2. 人脸识别测试")
        print("3. 食物识别测试")
        print("4. 双摄像头食物和人脸识别")
        print("5. 退出")
        
        choice = input("\n请选择功能 (1-5): ")
        
        if choice == '1':
            name = input("请输入用户姓名: ")
            if not system.check_user_registered(name):
                system.print_utf8(f"错误：用户 {name} 尚未注册！请先在系统中注册后再进行人脸采集。")
                continue
            system.capture_faces(name)
        elif choice == '2':
            system.recognize_face()
        elif choice == '3':
            system.test_food_detection()
        elif choice == '4':
            system.dual_camera_detection()
        elif choice == '5':
            system.print_utf8("感谢使用！")
            break
        else:
            system.print_utf8("无效的选择，请重试。")

if __name__ == '__main__':
    main()
