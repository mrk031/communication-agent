# 第一阶段：构建环境（安装依赖）
FROM python:3.10 as builder

WORKDIR /app

# 设置 pip 源（选用阿里云镜像加速，可选）
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# 复制依赖清单
COPY requirements.txt .

# 安装依赖到指定目录
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# 第二阶段：运行环境（最终镜像）
FROM python:3.10

WORKDIR /app

# 从第一阶段复制安装好的依赖库（这样镜像更小）
COPY --from=builder /install /usr/local

# 复制你的核心代码
COPY tasks_distribute.py intent_router.py /app/

# 暴露端口
EXPOSE 9450

# 启动命令
CMD ["python", "tasks_distribute.py"]
