FROM python:3.10 as builder

WORKDIR /app
#RUN pip install --upgrade pip
#安装需要的库
COPY requirements.txt /app/
RUN pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
#RUN apt-get update
#RUN apt-get install ffmpeg libsm6 libxext6  -y


# 两阶段构建 利用缓存
FROM python:3.10
WORKDIR /app
#RUN apt-get update
#RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# 将当前目录代码复制到容器
COPY . /app/
EXPOSE 9455
CMD ["python","langchainDirector.py"]
