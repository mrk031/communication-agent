import asyncio
import json
import time
import os
import logging
from datetime import datetime  # 新增：用于获取当前时间
from asyncio import Semaphore
from elasticsearch_dsl import Document, Text
from langchain_mcp_adapters.client import MultiServerMCPClient

# --- 新增：配置独立的文件日志记录器 ---
def setup_error_logger(name, log_file):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    if not os.path.exists('logs'):
        os.makedirs('logs')
    handler = logging.FileHandler(log_file, encoding='utf-8')
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.ERROR)
    logger.addHandler(handler)
    return logger

# 初始化专门记录 MCP 错误的 logger
mcp_error_logger = setup_error_logger('MCPErrorLog', 'logs/mcp_errors.log')
# 原有的控制台 logger
console_logger = logging.getLogger("MCPMonitor")

class MCPExperimentDocument(Document):
    interface_config = Text()
    class Index:
        name = "mcp_experiments"

class DeviceHealthMonitor:
    """
    硬件设备心跳监视器 (带工作时间段控制)
    """
    _instance = None
    _cache = {}
    _is_running = False
    _semaphore = Semaphore(50)

    # 配置文件路径
    CONFIG_FILE = "mcp_schedule.json"

    # 内存中的时间表 (默认值)
    _schedule_map = {
        "Default": {"start": 8, "end": 20}
    }

    @classmethod
    def load_schedule(cls):
        """从 JSON 文件加载时间配置"""
        if os.path.exists(cls.CONFIG_FILE):
            try:
                with open(cls.CONFIG_FILE, 'r', encoding='utf-8') as f:
                    cls._schedule_map = json.load(f)
                console_logger.info(f"已加载时间配置: {cls._schedule_map}")
            except Exception as e:
                console_logger.error(f"加载配置文件失败: {e}")
        else:
            # 文件不存在则创建默认
            cls.save_schedule()

    @classmethod
    def save_schedule(cls):
        """保存时间配置到 JSON 文件"""
        try:
            with open(cls.CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(cls._schedule_map, f, ensure_ascii=False, indent=2)
        except Exception as e:
            console_logger.error(f"保存配置文件失败: {e}")

    @classmethod
    def update_config(cls, model: str, start: int, end: int):
        """提供给 API 调用的更新接口"""
        cls._schedule_map[model] = {"start": start, "end": end}
        cls.save_schedule()
        return True

    @classmethod
    def get_model_schedule(cls, model: str):
        # 如果有特定配置，返回配置；否则返回 Default；再没有返回 8-20
        return cls._schedule_map.get(model, cls._schedule_map.get("Default", {"start": 8, "end": 20}))

    @classmethod
    def get_status(cls, hardware_id: str) -> dict:
        return cls._cache.get(hardware_id, {"status": "UNKNOWN", "error": "尚未检测"})

    @classmethod
    def get_all_status(cls) -> dict:
        return cls._cache

    @classmethod
    async def check_single_device(cls, hardware_id: str, mcp_address: str) -> bool:
        """
        检测单个设备
        Returns: True(成功), False(失败)
        """
        # 自动清洗地址：如果有 http:// 前缀则去掉，防止重复
        clean_address = mcp_address.replace("http://", "").replace("https://", "")

        check_config = {
            "heartbeat": {
                "transport": "streamable_http",
                "url": f"http://{clean_address}",
                "timeout": 5
            }
        }

        async with cls._semaphore:
            client = None
            try:
                client = MultiServerMCPClient(check_config)
                await client.get_tools()

                cls._cache[hardware_id] = {
                    "status": "ONLINE",
                    "mcp_address": mcp_address,
                    "last_check": time.strftime('%H:%M:%S'),
                    "error": None
                }
                return True  # <--- 修复点1：明确返回 True

            except Exception as e:
                # === 异常拆包逻辑 ===
                if hasattr(e, 'exceptions') and len(e.exceptions) > 0:
                    actual_error = e.exceptions[0]
                else:
                    actual_error = e

                # === 修复点2：字符串清洗逻辑 (去除 For more information...) ===
                raw_msg = str(actual_error)
                if "For more information check" in raw_msg:
                    raw_msg = raw_msg.split("For more information check")[0].strip()

                error_msg = f"{type(actual_error).__name__}: {raw_msg}"

                # 更新缓存
                cls._cache[hardware_id] = {
                    "status": "OFFLINE",
                    "mcp_address": mcp_address,
                    "last_check": time.strftime('%H:%M:%S'),
                    "error": error_msg
                }

                # 写入日志
                mcp_error_logger.error(f"HardwareID: {hardware_id} | Addr: {mcp_address} | Error: {error_msg}")

                return False  # <--- 修复点3：明确返回 False，以便 loop 函数统计

            finally:
                if client and hasattr(client, 'aclose'):
                    try:
                        await client.aclose()
                    except:
                        pass

    @staticmethod
    def is_device_working(start_h, end_h):
        current = datetime.now().hour
        # 简单逻辑：当前小时必须在 [start, end) 区间内
        return start_h <= current < end_h
    @classmethod
    async def loop(cls, interval=30):
        cls._is_running = True
        cls.load_schedule() # 加载配置
        console_logger.info("=== MCP监测启动 (配置模式: JSON动态配置) ===")

        while cls._is_running:
            try:
                start_time = time.time()

                # =======================================================
                # 【删除】原来在这里的全局时间判断代码 (if not cls._is_working_hours(): ...)
                # 直接进行 ES 查询
                # =======================================================

                # 1. 查询数据库获取设备列表
                s = MCPExperimentDocument.search() \
                    .query("term", interface_type__keyword="mcp") \
                    .source(["interface_config"]) \
                    .extra(size=1000)

                try:
                    response = s.execute()
                except Exception as e:
                    console_logger.error(f"ES查询失败: {e}")
                    await asyncio.sleep(5) # 查询失败稍作等待
                    continue

                tasks = []

                # 2. 遍历每个设备，独立判断时间
                for hit in response.hits:
                    try:
                        if isinstance(hit.interface_config, str):
                            config = json.loads(hit.interface_config)
                        else:
                            config = hit.interface_config

                        hw_id = config.get("hardware_id")
                        addr = config.get("mcp_address")
                        model = config.get("device_model", "Default")

                        # 读取动态配置
                        schedule = cls._schedule_map.get(model, cls._schedule_map.get("Default", {"start":8, "end":20}))
                        s_h = int(schedule["start"])
                        e_h = int(schedule["end"])

                        if hw_id and addr:
                            # === 新的逻辑：针对单台设备判断 ===
                            if cls.is_device_working(s_h, e_h):
                                # 在工作时间：加入检测任务
                                tasks.append(cls.check_single_device(hw_id, addr))
                            else:
                                # 不在工作时间：标记为 SLEEP
                                cls._cache[hw_id] = {
                                    "status": "SLEEP",
                                    "mcp_address": addr,
                                    "last_check": time.strftime('%H:%M:%S'),
                                    "error": f"非工作时间 ({s_h}:00-{e_h}:00)"
                                }

                    except Exception as e:
                        console_logger.error(f"配置解析失败: {e}")

                # 3. 执行检测
                if tasks:
                    await asyncio.gather(*tasks)

                # 4. 计算休眠时间
                elapsed = time.time() - start_time
                sleep_time = max(1.0, interval - elapsed)
                await asyncio.sleep(sleep_time)

            except Exception as e:
                console_logger.error(f"MCPMonitor 循环异常: {e}")
                await asyncio.sleep(5)

    @classmethod
    def stop(cls):
        cls._is_running = False
