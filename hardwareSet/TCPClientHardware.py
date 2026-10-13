import socket
import json


class HardwareClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port

    def send_string_to_server(self, message):
        # 创建一个TCP/IP套接字
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # 将消息转换为JSON格式并发送数据
            json_message = json.dumps(message)
            print(f"Sent: {json_message}")

            # 连接到服务器
            sock.connect((self.host, self.port))
            print(f"Connected to {self.host}:{self.port}")

            # 发送
            sock.sendall(json_message.encode('utf-8'))

            # # 暂停程序
            # input("按回车继续...")




