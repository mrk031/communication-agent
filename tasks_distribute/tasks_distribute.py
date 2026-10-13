# -*- coding: utf-8 -*-
import json
import copy
from typing import Optional, Dict, Any, List, TypedDict, AsyncGenerator, Tuple
from pydantic import BaseModel
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import os
import asyncio
import uuid
import requests
import aiohttp
import logging
import re
import math
import time
try:
    from .intent_router import classify_business_intents
    from .context_manager import select_recent_context
except ImportError:  # 兼容在 tasks_distribute 目录中直接启动脚本/uvicorn
    from intent_router import classify_business_intents
    from context_manager import select_recent_context
from fastapi.middleware.cors import CORSMiddleware
from ragflow_sdk import RAGFlow

# 设置日志记录
logging.basicConfig(level=logging.INFO)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 服务地址全部由环境变量注入，避免把内网拓扑和凭据写入仓库。
CTC_URL = os.getenv("CTC_URL", "http://127.0.0.1:9451/ctc_server")
DISCUSSION_URL = os.getenv("DISCUSSION_URL", "http://127.0.0.1:9452/discussion")
LLM_URL = os.getenv("LLM_URL", "http://127.0.0.1:9453/chat")
OCR_URL = os.getenv("OCR_URL", "http://127.0.0.1:9454/ocr_inference")
H5_URL = os.getenv("H5_URL", "http://127.0.0.1:9457/H5_server")
HardWare_URL = os.getenv("HARDWARE_URL", "http://127.0.0.1:9458/HardWare_server")
HardWareGetArgs_URL = os.getenv("HARDWARE_ARGS_URL", "http://127.0.0.1:9455/GetArgs")
HARDWARE_ARGS_TIMEOUT_SEC = float(os.getenv("HARDWARE_ARGS_TIMEOUT_SEC", "1.2"))
COURSE_EXPERIMENT_URL = os.getenv("COURSE_EXPERIMENT_URL", "http://127.0.0.1:8001/find_experiments")
IDEOLOGY_URL = os.getenv("IDEOLOGY_URL", "http://127.0.0.1:8005/search")

# 新增：历史记录服务地址 (根据前端8082端口推断)
HISTORY_URL = os.getenv("HISTORY_URL", "http://127.0.0.1:8082/ctc_agent/conversations/messages")
HISTORY_TOKEN_BUDGET = int(os.getenv("HISTORY_TOKEN_BUDGET", "1600"))
HISTORY_MAX_MESSAGES = int(os.getenv("HISTORY_MAX_MESSAGES", "10"))
HISTORY_MAX_MESSAGE_CHARS = int(os.getenv("HISTORY_MAX_MESSAGE_CHARS", "2000"))

# SDK集成: RagFlow SDK 配置
RAGFLOW_SDK_BASE_URL = os.getenv("RAGFLOW_BASE_URL", "http://127.0.0.1:9380")
RAGFLOW_API_KEY = os.getenv("RAGFLOW_API_KEY", "")
RAGFLOW_TARGET_DATASET_NAME = os.getenv("RAGFLOW_DATASET_NAME", "通原")

# Gold-dev 冻结配置（e02）：RAGFlow 先返回 Top-10 候选，再取前4条进入LLM上下文。
# top_k 是RAGFlow内部向量候选规模，不等于最终返回或送入LLM的数量。
RAGFLOW_SIMILARITY_THRESHOLD = 0.2
RAGFLOW_VECTOR_WEIGHT = 0.6
RAGFLOW_CANDIDATE_K = 10
RAGFLOW_CONTEXT_K = 4
RAGFLOW_EMBEDDING_TOP_K = 1024

# rules: 普通问题不再等待一次完整LLM分类；llm: 保留旧链路用于快速回滚。
INTENT_ROUTER_MODE = os.getenv("INTENT_ROUTER_MODE", "rules").strip().lower()

target_dataset_ids_for_ragflow: List[str] = []
ragflow_client: Optional[RAGFlow] = None


@app.on_event("startup")
async def startup_event():
    global ragflow_client, target_dataset_ids_for_ragflow
    logging.info("Application startup: Initializing RAGFlow client...")
    if not RAGFLOW_API_KEY:
        logging.error("RAGFlow API Key 未有效配置，SDK客户端初始化跳过。")
        return

    try:
        ragflow_client = RAGFlow(api_key=RAGFLOW_API_KEY, base_url=RAGFLOW_SDK_BASE_URL)
        logging.info(f"RAGFlow SDK客户端实例已创建，连接到: {RAGFLOW_SDK_BASE_URL}")

        logging.info(f"尝试获取数据集 '{RAGFLOW_TARGET_DATASET_NAME}' 的ID...")
        datasets_found = await asyncio.to_thread(
            ragflow_client.list_datasets,
            name=RAGFLOW_TARGET_DATASET_NAME
        )

        if datasets_found and hasattr(datasets_found[0], 'id'):
            target_dataset_ids_for_ragflow = [str(datasets_found[0].id)]
            logging.info(f"成功获取到数据集 '{RAGFLOW_TARGET_DATASET_NAME}' 的ID: {target_dataset_ids_for_ragflow}")
        else:
            logging.error(f"未能找到名为 '{RAGFLOW_TARGET_DATASET_NAME}' 的数据集或返回对象结构不符合预期。")
            target_dataset_ids_for_ragflow = []

    except Exception as e:
        logging.error(f"初始化RAGFlow SDK客户端或获取数据集ID时发生错误: {e}", exc_info=True)
        ragflow_client = None
        target_dataset_ids_for_ragflow = []


templates = Jinja2Templates(directory="templates")

# 临时文件保存目录
TEMP_FILE_DIR = "temp_files"
os.makedirs(TEMP_FILE_DIR, exist_ok=True)


# 数据模型定义
class Attachment(BaseModel): id: str; type: str


class ChatMessage(BaseModel): chat_id: int; content: str; attachments: List[Attachment] = []


class ChatMessage_H5(BaseModel): chat_id: int; content: str; H5_List: List; attachments: List[Attachment] = []


class NextQuestionRequest(BaseModel):
    question: str
    answer: str
    model: str = "GPT"


# 业务类型模板
business_types_template = {
    '问题内容': '', '通信原理题目库': 0, '通信原理课堂教师答疑记录': 0,
    '大模型概念回答': 0, '通信原理课堂交互式案例': 0, '通信原理课堂在线硬件实验': 0,
    '其他课程实验案例': 0,
    '思政教育案例': 0  # <--- 新增
}

# 显示前端页面
@app.get("/", response_class=HTMLResponse)
async def get_frontend_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# 保存上传的图片文件到本地
async def save_image_locally(image_file: UploadFile) -> str:
    try:
        file_ext = image_file.filename.split('.')[-1]
        unique_filename = str(uuid.uuid4()) + "." + file_ext
        temp_file_path = os.path.join(TEMP_FILE_DIR, unique_filename)
        with open(temp_file_path, "wb") as buffer:
            content = await image_file.read()
            buffer.write(content)
        logging.info(f"图片已保存到临时文件: {temp_file_path}")
        return temp_file_path
    except Exception as e:
        logging.error(f"保存图片时发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"保存图片时发生错误: {str(e)}")


# OCR识别相关函数
async def get_ocr_text_async_wrapper(file_path: str) -> str:
    return await asyncio.to_thread(get_ocr_text_sync, file_path)


def get_ocr_text_sync(file_path: str) -> str:
    try:
        with open(file_path, 'rb') as file:
            response = requests.post(OCR_URL, files={"file": file})
        response.raise_for_status()
        ocr_result = response.json()
        logging.info(f"OCR完成 ,返回结果: {ocr_result}")
        return ocr_result["result"]
    except Exception as e:
        logging.error(f"OCR处理发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OCR处理发生错误: {str(e)}")


# -------------------- 核心工具：处理流式 LLM 请求 --------------------

async def get_llm_response_full(payload: dict) -> str:
    full_content = ""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(LLM_URL, json=payload) as response:
                if response.status != 200:
                    logging.error(f"LLM request failed with status {response.status}")
                    return ""

                # 逐行读取 SSE 数据
                while True:
                    line_bytes = await response.content.readline()
                    if not line_bytes:
                        break
                    line_text = line_bytes.decode('utf-8').rstrip('\n')
                    logging.debug(f"Received line: {line_text}")  # 添加调试日志

                    if line_text.startswith("data: "):
                        data_str = line_text[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data_json = json.loads(data_str)
                            if "choices" in data_json:
                                delta = data_json["choices"][0]["delta"].get("content", "")
                                full_content += delta
                        except json.JSONDecodeError as e:
                            logging.error(f"JSON decode error: {e}, line: {line_text}")
                        except Exception:
                            pass
        return full_content
    except Exception as e:
        logging.error(f"Error getting full LLM response: {e}", exc_info=True)
        return ""

# -------------------- 获取历史对话记录 --------------------

async def get_chat_history(conv_id: str) -> Tuple[str, List[Dict[str, str]], str]:
    formatted_history = ""
    messages_list = []
    user_only_history = ""
    if not conv_id:
        return formatted_history, messages_list, user_only_history

    try:
        async with aiohttp.ClientSession() as session:
            url = f"{HISTORY_URL}?conv_id={conv_id}"
            async with session.get(url) as response:
                if response.status == 200:
                    res_json = await response.json()
                    if res_json.get("success") and res_json.get("data"):
                        messages = res_json.get("data", [])
                        recent_messages, context_stats = select_recent_context(
                            messages,
                            token_budget=HISTORY_TOKEN_BUDGET,
                            max_messages=HISTORY_MAX_MESSAGES,
                            max_message_chars=HISTORY_MAX_MESSAGE_CHARS,
                        )
                        logging.info("历史上下文预算结果: %s", context_stats)

                        history_builder = []
                        user_queries = []
                        for msg in recent_messages:
                            role = msg.get("role", "assistant")
                            content = msg.get("content", "")

                            if content:
                                role_str = "User" if role == "user" else "Assistant"
                                history_builder.append(f"{role_str}: {content}")
                                messages_list.append({"role": role, "content": content})
                                # 仅收集用户的提问内容
                                if role == "user":
                                    user_queries.append(content)

                        if history_builder:
                            formatted_history = "\n".join(history_builder)
                        if user_queries:
                            user_only_history = " ".join(user_queries)  # 合并为一个长字符串供匹配使用
    except Exception as e:
        logging.error(f"获取历史记录失败: {e}")
    return formatted_history, messages_list, user_only_history


# -------------------- 业务逻辑处理函数 --------------------

async def auto_select_business_type(question: str, model_type: str) -> dict:
    """选择辅助业务面板；默认走保守规则，旧LLM路由可通过环境变量回滚。"""

    if INTENT_ROUTER_MODE != "llm":
        return classify_business_intents(question)

    prompt = (
        f"{question}\n根据上述的问题选择最合适的业务类型，并按格式返回对应的数字：\n返回内容有且只有如下格式内容，不要添加任何多余文字、字符：\n"
        "{'问题内容':'xxx','通信原理题目库': 1, '通信原理课堂教师答疑记录': 1, '大模型概念回答': 1,'通信原理课堂交互式案例': 1,'通信原理课堂在线硬件实验': 1,'其他课程实验案例': 1,'思政教育案例': 1}\n"
        "我们有多个可用的业务类型：\n1. 通信原理题目库（里面有大量的通信原理相关习题）\n2. 通信原理课堂教师答疑记录\n3. 大模型概念回答（用于概要介绍通信原理相关概念的大模型）\n"
        "4. 通信原理课堂交互式案例(可以在线进行相关的软件仿真实验)\n"
        "5. 通信原理课堂在线硬件实验(判定标准：只要用户问题中包含'做实验'、'找实验'、'实验演示'、'硬件'等意图，请务必将此项标记为 1)\n"
        "6. 其他课程实验案例(判定标准：只要用户问题中包含'做实验'、'找实验'、'实验演示'等意图，无论是否属于通信原理课程，或者属于其他课程，请务必将此项标记为 1)\n"
        "7. 思政教育案例(判定标准：只要用户问题中包含'科学家'、'历史'、'发展史'、'先烈'、'北邮名师'、'榜样'、'爱国'、'技术突破'、'精神'，或者询问具体人物如'张平'、'马可尼'、'香农'的故事，请务必将此项标记为 1)\n"
        "请分析问题，并根据问题的内容判断哪些业务类型相关。\n对于每个业务类型，如果问题相关，请返回 1；如果不相关，请返回 0。\n请以字符串的格式返回，字典中每个业务类型的值为 1 或 0，表示该业务类型是否与问题相关。\n"
    )

    payload = {"message": prompt, "model": model_type}
    logging.info("--- Intent Recognition Start ---")

    full_text = await get_llm_response_full(payload)
    logging.info(f"Intent Raw Response: {full_text}")

    try:
        clean_text = full_text.replace("```json", "").replace("```", "").strip()
        str_data = clean_text.replace("'", '"')
        content = json.loads(str_data)

        if '问题内容' not in content or not content['问题内容']:
            content['问题内容'] = question
        return content
    except Exception as e:
        logging.error(f"Intent JSON parse error: {e}")
        return {'问题内容': question, '通信原理题目库': 1, '大模型概念回答': 1}


# -------------------- 独立的后台任务函数 --------------------

async def fetch_other_course_experiments(question: str, model_type: str) -> dict:
    """
    查询其他课程实验案例
    """
    payload = {"question": question, "model_type": model_type}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(COURSE_EXPERIMENT_URL, json=payload) as response:
                if response.status != 200:
                    return {"content": "查询实验案例时服务调用失败", "list": []}

                res_json = await response.json()
                course_name = res_json.get("course_name", "")

                if not res_json.get("success", False):
                    return {"content": res_json.get("message", "未找到相关实验案例"), "list": []}

                experiments = res_json.get("experiments", [])
                if not experiments:
                    return {"content": f"未找到相关实验案例", "list": []}

                content_str = f"根据您的提问，为您找到以下实验案例：\n"
                for i, exp in enumerate(experiments, 1):
                    content_str += f"{i}. {exp.get('key_word', '')}\n"

                return {
                    "content": content_str,
                    "list": [{"实验案例": exp.get("key_word", ""), "链接": exp.get("url", "")} for exp in experiments]
                }
    except Exception as e:
        logging.error(f"Other course experiment error: {e}")
        return {"content": "查询实验案例时发生异常", "list": []}

async def fetch_ideology_cases(question: str, model_type: str) -> dict:
    """
    查询思政教育案例 (仅返回 title 和 url)
    """
    payload = {"prompt": question, "model_type": model_type}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(IDEOLOGY_URL, json=payload) as response:
                if response.status != 200:
                    return {"content": "查询思政案例时服务调用失败", "list": []}

                res_json = await response.json()

                if not res_json.get("success", False):
                    return {"content": res_json.get("message", "未找到相关思政案例"), "list": []}

                cases = res_json.get("cases", [])
                if not cases:
                    return {"content": "未找到相关思政案例", "list": []}

                content_str = "根据您的提问，为您找到以下思政与历史发展案例：\n"
                for i, case in enumerate(cases, 1):
                    # 精简回复文案，只留标题
                    content_str += f"{i}. {case.get('title', '')}\n"

                return {
                    "content": content_str,
                    "list": [
                        {
                            "案例名称": case.get("title", ""),
                            "链接": case.get("url", "")
                        } for case in cases
                    ]
                }
    except Exception as e:
        logging.error(f"Ideology cases fetch error: {e}")
        return {"content": "查询思政案例时发生异常", "list": []}


def fetch_question_bank_sync(question: str) -> dict:
    try:
        response = requests.post(CTC_URL, json={"message": question})
        if response.status_code == 200:
            result = response.json()
            return {"content": result.get("content", ""), "list": result.get("list", [])}
    except Exception:
        pass
    return {"content": "通信原理题库请求失败", "list": []}


def fetch_qa_record_sync(question: str) -> dict:
    try:
        response = requests.post(DISCUSSION_URL, json={"message": question})
        if response.status_code == 200:
            result = response.json()
            return {"content": result.get("content", ""), "list": result.get("list", [])}
    except Exception:
        pass
    return {"content": "教师问答记录请求失败", "list": []}


def fetch_interactive_stories_sync(question: str) -> dict:
    try:
        response = requests.post(H5_URL, json={"message": question})
        if response.status_code == 200:
            result = response.json()
            # 修正：获取 'H5_List' 并赋值给前端需要的 'list' 键
            return {"content": result.get("content", ""), "list": result.get("H5_List", [])}
    except Exception:
        pass
    return {"content": "交互式案例请求失败", "list": []}


# -------------------- 硬件实验：上下文注意力选择器 V2 --------------------

# Query：当前用户问题；Key：最近历史中包含实验实体的消息片段；Value：对应实验名称。
# V2 修复：
# 1. 比较/对比类问题从注意力分布中取 top2/top3，不再强行 top1。
# 2. 增加父子实验抑制，避免 QPSK/DPSK 被泛化成 PSK。
EXPERIMENT_ALIASES: Dict[str, List[str]] = {
    "AM": ["AM", "调幅", "幅度调制", "普通调幅"],
    "DSB": ["DSB", "双边带"],
    "SSB": ["SSB", "单边带"],
    "ASK": ["ASK", "2ASK", "幅移键控", "二进制幅移键控"],
    "FSK": ["FSK", "2FSK", "频移键控", "二进制频移键控"],
    "DPSK": ["DPSK", "差分相移键控", "差分调制", "差分编码"],
    "QPSK": ["QPSK", "四相相移键控", "四相调制"],
    "PSK": ["PSK", "2PSK", "二进制相移键控", "相移键控"],
    "PCM": ["PCM", "脉冲编码调制"],
    "HDB3": ["HDB3"],
    "AMI": ["AMI"],
    "均衡器": ["均衡器", "信道均衡", "均衡实验"],
    "抽样定理": ["抽样定理", "采样定理", "抽样实验", "采样实验"],
}

# 出现更具体实验时，抑制泛化父类。
EXPERIMENT_PARENT_SUPPRESSIONS: Dict[str, List[str]] = {
    "QPSK": ["PSK"],
    "DPSK": ["PSK"],
}

# 多目标任务门控：只判断“任务是否需要多个对象”，不做“这个/那个”指代规则。
MULTI_TARGET_TASK_CUES = [
    "比较", "对比", "区别", "差异", "异同", "不同", "相同", "优缺点", "哪个好",
    "两种", "两个", "二者", "二者之间", "2种", "2个",
    "三种", "三个", "3种", "3个",
    "分别", "都", "全部", "所有", "一起"
]

# 原始实验名识别：解决“星座图实验”这类数据库中真实存在、但没有写进 EXPERIMENT_ALIASES 的实验。
# 这里不做语义扩展，不会把“星座图实验”扩成 QPSK/PSK。
RAW_EXPERIMENT_ACTION_WORDS = [
    "帮我找", "给我找", "我要做", "我想做", "帮我打开", "给我打开",
    "打开", "进入", "找一下", "查一下", "有没有", "请问", "能不能",
    "可以不可以", "相关的", "对应的"
]

RAW_EXPERIMENT_SUFFIXES = ["实验", "硬件", "演示", "实训", "操作"]


def looks_like_raw_experiment_name(question: str) -> bool:
    """
    判断用户是不是直接在问某个实验名。
    例如：星座图实验、眼图实验、码型变换实验、抽样定理实验。
    不依赖 EXPERIMENT_ALIASES，因此新增实验不用改代码。
    """
    q = (question or "").strip()
    if not q:
        return False

    # 用户明确说“xxx实验/xxx硬件/xxx演示”，就认为是原始实验名查询。
    if any(suffix in q for suffix in RAW_EXPERIMENT_SUFFIXES):
        return True

    return False


def clean_raw_experiment_name(question: str) -> str:
    """
    清洗口语化前后缀，保留用户原始实验名。
    注意：不做语义扩展、不做实验映射。
    """
    q = (question or "").strip()

    for word in RAW_EXPERIMENT_ACTION_WORDS:
        q = q.replace(word, "")

    # 去掉常见标点和空白。
    q = re.sub(r"[\s，。！？?；;：:、]+$", "", q)
    q = re.sub(r"^[\s，。！？?；;：:、]+", "", q)

    return q or (question or "").strip()




def should_force_online_hardware_experiment(question: str) -> bool:
    """
    硬规则：只要用户问题里出现“实验/硬件/演示”等实验意图，
    就强制调用“通信原理课堂在线硬件实验”。

    目的：不再完全依赖大模型意图识别，避免 LLM 把“星座图实验”误判为
    交互式案例=1、在线硬件实验=0，导致 9458 根本没有被调用。
    """
    q = (question or "").strip()
    if not q:
        return False

    force_cues = [
        "实验", "硬件", "实训", "演示", "实验演示",
        "做实验", "找实验", "打开实验", "进入实验",
        "在线实验", "硬件实验", "实验入口", "实验链接",
        "开始实验", "启动实验", "操作实验", "实验平台"
    ]
    return any(cue in q for cue in force_cues)

def normalize_for_attention(text: str) -> str:
    if not text:
        return ""
    text = str(text).upper()
    text = re.sub(r"\s+", "", text)
    return text


def char_ngram_vector(text: str, n: int = 2) -> Dict[str, float]:
    text = normalize_for_attention(text)
    if not text:
        return {}
    if len(text) <= n:
        return {text: 1.0}
    vec: Dict[str, float] = {}
    for i in range(len(text) - n + 1):
        gram = text[i:i + n]
        vec[gram] = vec.get(gram, 0.0) + 1.0
    return vec


def cosine_sim_text(a: str, b: str) -> float:
    va = char_ngram_vector(a)
    vb = char_ngram_vector(b)
    if not va or not vb:
        return 0.0
    common = set(va) & set(vb)
    dot = sum(va[k] * vb[k] for k in common)
    na = math.sqrt(sum(v * v for v in va.values()))
    nb = math.sqrt(sum(v * v for v in vb.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def softmax(scores: Dict[str, float], temperature: float = 0.42) -> Dict[str, float]:
    if not scores:
        return {}
    max_score = max(scores.values())
    exp_scores = {
        key: math.exp((score - max_score) / max(temperature, 1e-6))
        for key, score in scores.items()
    }
    total = sum(exp_scores.values()) or 1.0
    return {key: value / total for key, value in exp_scores.items()}


def suppress_parent_experiments(experiments: List[str]) -> List[str]:
    ordered = list(dict.fromkeys(experiments))
    hit_set = set(ordered)
    suppressed = set()
    for child, parents in EXPERIMENT_PARENT_SUPPRESSIONS.items():
        if child in hit_set:
            suppressed.update(parents)
    return [exp for exp in ordered if exp not in suppressed]


def extract_experiments_by_entity(text: str) -> List[str]:
    """
    只识别明确实验实体；通过“长别名优先 + 父子抑制”避免 QPSK/DPSK 被识别成 PSK。
    """
    if not text:
        return []

    normalized = normalize_for_attention(text)
    raw_hits: List[str] = []
    matched_by_exp = set()

    alias_rows = []
    for exp_name, aliases in EXPERIMENT_ALIASES.items():
        for alias in aliases:
            alias_rows.append((exp_name, alias, normalize_for_attention(alias)))
    alias_rows.sort(key=lambda x: len(x[2]), reverse=True)

    for exp_name, alias, alias_norm in alias_rows:
        if not alias_norm or exp_name in matched_by_exp:
            continue

        if re.fullmatch(r"[A-Z0-9]+", alias_norm):
            pattern = rf"(?<![A-Z0-9]){re.escape(alias_norm)}(?![A-Z0-9])"
            if re.search(pattern, normalized):
                raw_hits.append(exp_name)
                matched_by_exp.add(exp_name)
        else:
            if alias_norm in normalized:
                raw_hits.append(exp_name)
                matched_by_exp.add(exp_name)

    return suppress_parent_experiments(raw_hits)


def build_attention_memory(history_messages: List[Dict[str, str]], max_messages: int = 16) -> List[Dict[str, Any]]:
    if not history_messages:
        return []

    recent = history_messages[-max_messages:]
    memory: List[Dict[str, Any]] = []
    for idx, msg in enumerate(recent):
        content = msg.get("content", "") or ""
        if not content:
            continue
        experiments = extract_experiments_by_entity(content)
        if not experiments:
            continue
        role = msg.get("role", "user")
        for exp in experiments:
            memory.append({
                "key_text": content,
                "value": exp,
                "turn_pos": idx + 1,
                "total_turns": len(recent),
                "role": role,
            })
    return memory


def infer_requested_target_count(question: str, available_count: int) -> int:
    """
    判断当前 Query 需要取几个 Value。
    这不是“这个/那个”指代匹配，而是比较任务天然需要多个实验。
    """
    if available_count <= 1:
        return max(available_count, 1)

    q = question or ""
    if any(x in q for x in ["三种", "三个", "3种", "3个"]):
        return min(3, available_count)
    if any(x in q for x in ["两种", "两个", "二者", "2种", "2个"]):
        return min(2, available_count)
    if any(cue in q for cue in MULTI_TARGET_TASK_CUES):
        return min(2, available_count)
    return 1


def select_by_attention_weights(weights: Dict[str, float], requested_count: int, min_keep_weight: float = 0.12) -> List[str]:
    if not weights:
        return []
    sorted_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    if requested_count <= 1:
        return [sorted_items[0][0]]

    selected: List[str] = []
    for idx, (exp, weight) in enumerate(sorted_items):
        if idx < requested_count and (weight >= min_keep_weight or idx == 0):
            selected.append(exp)
    return selected or [sorted_items[0][0]]


def resolve_hardware_experiment_by_attention(
        question: str,
        history_messages: List[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    使用上下文注意力选择硬件实验。
    V2：比较类问题保留多个高权重实验；QPSK/DPSK 会抑制泛化 PSK。
    """
    if history_messages is None:
        history_messages = []

    current_entities = extract_experiments_by_entity(question)
    memory = build_attention_memory(history_messages)

    if current_entities:
        requested_count = infer_requested_target_count(question, len(current_entities))
        selected = current_entities[:requested_count]
        weights = {exp: round(1.0 / len(selected), 4) for exp in selected} if selected else {}
        return {
            "mode": "attention_current_query_multi" if len(selected) > 1 else "attention_current_query",
            "selected_experiments": selected,
            "search_query": " ".join(selected),
            "attention_weights": weights,
            "memory_size": len(memory),
            "confidence": max(weights.values()) if weights else 0.0,
            "requested_count": requested_count,
            "reason": "当前问题中出现明确实验实体；比较类问题保留多个实验。"
        }

    scores_by_exp: Dict[str, float] = {}
    detail_rows: List[Dict[str, Any]] = []

    for item in memory:
        exp = item["value"]
        key_text = item["key_text"]
        recency = item["turn_pos"] / max(item["total_turns"], 1)
        role_weight = 1.0 if item.get("role") == "user" else 0.72

        semantic_score = cosine_sim_text(question, key_text)
        experiment_score = cosine_sim_text(question, exp)
        raw_score = (
            0.50 * semantic_score +
            0.34 * recency +
            0.10 * experiment_score +
            0.06 * role_weight
        )

        old_score = scores_by_exp.get(exp, 0.0)
        scores_by_exp[exp] = max(old_score, raw_score) + 0.04 * old_score
        detail_rows.append({
            "experiment": exp,
            "semantic": round(semantic_score, 4),
            "recency": round(recency, 4),
            "raw_score": round(raw_score, 4),
            "key_preview": key_text[:60]
        })

    # 分数层面再次抑制父类 PSK，防止历史里“相移键控”泛化项压过 QPSK/DPSK。
    for child, parents in EXPERIMENT_PARENT_SUPPRESSIONS.items():
        if child in scores_by_exp:
            for parent in parents:
                if parent in scores_by_exp:
                    scores_by_exp[parent] *= 0.35

    weights = softmax(scores_by_exp)
    if weights:
        sorted_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        requested_count = infer_requested_target_count(question, len(sorted_items))
        selected = select_by_attention_weights(weights, requested_count)
        return {
            "mode": "attention_history_memory_multi" if len(selected) > 1 else "attention_history_memory",
            "selected_experiments": selected,
            "search_query": " ".join(selected),
            "attention_weights": {k: round(v, 4) for k, v in sorted_items[:5]},
            "attention_details": sorted(detail_rows, key=lambda x: x["raw_score"], reverse=True)[:8],
            "memory_size": len(memory),
            "confidence": round(sum(weights.get(exp, 0.0) for exp in selected), 4),
            "requested_count": requested_count,
            "reason": "当前问题未出现实验实体，使用 Query-Key-Value 注意力从历史实验记忆中选择；比较类问题保留多个高权重实验。"
        }

    # 如果当前问题没有命中手写别名，但用户明显是在问“某某实验”，
    # 就把原始实验名直接作为目标实验传给 9458。
    # 例如：“星座图实验”不在 EXPERIMENT_ALIASES 里，也应该原样检索“星座图实验”。
    if looks_like_raw_experiment_name(question):
        raw_exp_name = clean_raw_experiment_name(question)
        return {
            "mode": "raw_experiment_name_query",
            "selected_experiments": [raw_exp_name],
            "search_query": raw_exp_name,
            "attention_weights": {raw_exp_name: 1.0},
            "memory_size": len(memory),
            "confidence": 1.0,
            "requested_count": 1,
            "reason": "当前问题像是直接询问某个实验名；未命中手写别名表时，保留原始实验名交给 9458 精确检索。"
        }

    return {
        "mode": "attention_no_memory",
        "selected_experiments": [],
        "search_query": question,
        "attention_weights": {},
        "memory_size": 0,
        "confidence": 0.0,
        "requested_count": 1,
        "reason": "没有可用历史实验记忆，回退为当前问题检索。"
    }



async def fetch_hardware_args_with_timeout(
        session: aiohttp.ClientSession,
        question: str,
        search_query: str,
        attention_result: Dict[str, Any],
        timeout_sec: float = HARDWARE_ARGS_TIMEOUT_SEC
) -> Dict[str, Any]:
    """
    调用 9455/GetArgs 获取硬件实验参数。

    优化点：
    1. 9455 只负责“自然语言 -> 硬件实验参数”；
    2. 给 9455 设置短超时，避免它拖慢 9458 返回实验链接；
    3. 9455 超时、报错、返回异常状态码时，直接返回空参数 {}；
    4. 简化传给 9455 的文本，只保留用户问题和实验名称，减少解析耗时。
    """
    start_time = asyncio.get_running_loop().time()

    try:
        args_question = (
            f"用户问题：{question}\n"
            f"实验名称：{search_query}"
        )

        timeout = aiohttp.ClientTimeout(total=timeout_sec)

        async with session.post(
                HardWareGetArgs_URL,
                json={"message": args_question},
                timeout=timeout
        ) as response:
            cost_time = asyncio.get_running_loop().time() - start_time

            if response.status != 200:
                logging.warning(
                    f"9455 GetArgs 状态码异常: status={response.status}, "
                    f"cost={cost_time:.3f}s"
                )
                return {}

            args_json = await response.json()
            args_data = args_json.get("HardWareArgs", {})

            logging.info(
                f"9455 GetArgs 调用成功: "
                f"cost={cost_time:.3f}s, args={args_data}"
            )
            return args_data if isinstance(args_data, dict) else {}

    except asyncio.TimeoutError:
        cost_time = asyncio.get_running_loop().time() - start_time
        logging.warning(
            f"9455 GetArgs 超时，已跳过参数解析: "
            f"timeout={timeout_sec}s, cost={cost_time:.3f}s"
        )
        return {}

    except Exception as e:
        cost_time = asyncio.get_running_loop().time() - start_time
        logging.error(
            f"9455 GetArgs 调用失败，已降级为空参数: "
            f"cost={cost_time:.3f}s, error={e}",
            exc_info=True
        )
        return {}


async def fetch_hardware_experiments_async(
        question: str,
        model_type: str = "DeepSeek",
        history_messages: List[Dict[str, str]] = None,
        conv_id: str = ""
) -> dict:
    """
    注意力版硬件实验选择：
    1. 不再汇总 user_only_history 中的所有实验关键词。
    2. 用当前问题作为 Query、历史实验片段作为 Key、实验名作为 Value。
    3. 只把注意力最高的实验发给 9458 端口，避免历史实验全部被调用。
    """
    try:
        if history_messages is None:
            history_messages = []

        attention_result = resolve_hardware_experiment_by_attention(question, history_messages)
        selected_experiments = attention_result.get("selected_experiments", [])
        search_query = attention_result.get("search_query") or question

        # V2：比较/对比类问题允许返回多个实验；普通追问仍然只返回 top1。
        # 如果是 raw_experiment_name_query，例如“星座图实验”，selected_experiments 会直接是 ["星座图实验"]。
        if selected_experiments:
            top_k = max(1, min(len(selected_experiments), 3))
        else:
            top_k = max(1, min(int(attention_result.get("requested_count", 1) or 1), 3))

        target_experiment = selected_experiments[0] if selected_experiments else ""
        target_experiments = selected_experiments

        logging.info(
            "[硬件实验注意力选择] "
            f"conv_id={conv_id}, "
            f"mode={attention_result.get('mode')}, "
            f"query='{question}', "
            f"selected={selected_experiments}, "
            f"search_query='{search_query}', "
            f"weights={attention_result.get('attention_weights')}, "
            f"confidence={attention_result.get('confidence')}"
        )

        async with aiohttp.ClientSession() as session:
            # 9458：硬件实验匹配/链接返回，是主流程，必须尽量保证成功。
            task_match = session.post(HardWare_URL, json={
                "message": search_query,
                "target_experiment": target_experiment,
                "target_experiments": target_experiments,
                "top_k": top_k,
                "attention": attention_result,
            })

            # 9455：实验参数解析，是辅助流程。
            # 加短超时：慢了就返回 {}，不再长时间拖住 9458 的实验链接结果。
            task_args = fetch_hardware_args_with_timeout(
                session=session,
                question=question,
                search_query=search_query,
                attention_result=attention_result,
                timeout_sec=HARDWARE_ARGS_TIMEOUT_SEC
            )

            resp_match, args_data = await asyncio.gather(
                task_match,
                task_args,
                return_exceptions=True
            )

            match_result = {}
            if not isinstance(resp_match, Exception) and resp_match.status == 200:
                match_result = await resp_match.json()
            else:
                logging.error(f"9458 Hardware match request failed: {resp_match}")

            if isinstance(args_data, Exception):
                logging.error(f"9455 Hardware args request failed: {args_data}")
                args_data = {}

            logging.info(
                "[硬件实验接口返回] "
                f"9458_list_count={len(match_result.get('H5_List', []))}, "
                f"9455_args={args_data}"
            )

            return {
                "content": match_result.get("content", ""),
                "list": match_result.get("H5_List", []),
                "HardWareArgs": args_data,
                "HardwareAttention": attention_result,
                "HardwareRankDebug": match_result.get("RankDebug", [])
            }

    except Exception as e:
        logging.error(f"Hardware experiment fetch error: {e}", exc_info=True)
        return {
            "content": "硬件实验请求失败",
            "list": [],
            "HardWareArgs": {},
            "HardwareAttention": {}
        }


# -------------------- 主流式生成器 (已增加记忆功能) --------------------

async def stream_concept_answer_generator(question: str, model: str, rag_chunks: str,
                                          history_messages: List[Dict] = None,
                                          is_reasoning: bool = False) -> AsyncGenerator[Dict, None]:
    """负责调用大模型并逐字yield token (概念回答)，支持历史记忆、深度思考(兼容Tag和字段模式)"""

    if history_messages is None:
        history_messages = []

    # --- 改进后的指令：增强指代一致性约束 ---
    sys_instruction = (
        "你是一个专业的通信原理助手。请严格遵守以下规则回答：\n"
        "1. 上下文优先原则：如果用户要求'比较'或指代'上面提到的'，你必须优先从历史对话中提取最近的通信概念进行回答。\n"
        "2. 简洁性：在保证准确的前提下，用较少的文字回答。\n"
        "3. 领域锁定：所有回答必须锁定在通信领域内。"
    )

    # 【核心修复】：将 RAG 知识库内容放入最后一轮的用户提问中，并添加明确的边界和指令
    final_user_content = question
    if rag_chunks:
        final_user_content = (
            f"【参考知识库】\n"
            f"{rag_chunks}\n\n"
            f"【用户提问】\n"
            f"{question}\n\n"
            f"要求：请结合参考知识库在独立的段落中给出专业、准确且精简的最终解答。"
        )
    # 请先结合参考知识库进行思考分析；思考结束后，请在独立的段落中给出专业、准确且精简的最终解答。
    # --- 1. 标准化多轮对话 Messages 数组组装 ---
    messages = [{"role": "system", "content": sys_instruction}]
    messages.extend(history_messages)
    messages.append({"role": "user", "content": final_user_content})

    # --- 2. 兼容处理老式 Payload 结构 ---
    prompt_fallback = f"{sys_instruction}\n\n{final_user_content}"

    # Payload 增加 messages 数组字段
    payload = {
        "messages": messages,
        "message": prompt_fallback,  # 兼容保留
        "model": model,
        "is_reasoning": is_reasoning
    }

    logging.info(f"=== LLM请求参数 ===")
    logging.info(f"模型: {model}")
    logging.info(f"是否深度思考: {is_reasoning}")
    logging.info(f"本次请求包含的历史轮数: {len(history_messages)}")

    # === 新增：思考标签状态标记，用于兼容 Ollama/vLLM 的 <think> 格式 ===
    is_in_thinking_tag = False

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(LLM_URL, json=payload) as response:
                if response.status != 200:
                    error_msg = await response.text()
                    logging.error(f"LLM Error: {response.status} - {error_msg}")
                    yield {"type": "token", "content": f"Error: LLM service returned {response.status}"}
                    return

                logging.info(f"LLM Connected. Streaming starts. Model: {model}, Reasoning: {is_reasoning}")

                async for line in response.content:
                    line_text = line.decode('utf-8').strip()
                    if line_text.startswith("data: "):
                        data_str = line_text[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data_json = json.loads(data_str)
                            logging.debug(f"原始LLM响应: {data_str[:100]}...")

                            if "choices" in data_json:
                                delta = data_json["choices"][0]["delta"]

                                # --- 兼容模式 1: 标准 API (DeepSeek 官方格式) ---
                                reasoning_content = delta.get("reasoning_content", "")
                                if reasoning_content:
                                    logging.info(f"标准API思考内容: {reasoning_content[:50]}...")
                                    yield {"type": "reasoning_token", "content": reasoning_content}
                                    continue  # 如果有专用字段，优先使用

                                # --- 兼容模式 2: 标签解析 (<think> 格式) ---
                                content = delta.get("content", "")
                                if not content:
                                    continue

                                # 检测标签切换状态
                                if "<think>" in content:
                                    is_in_thinking_tag = True
                                    content = content.replace("<think>", "")  # 移除标签本身
                                    logging.info("检测到<think>标签，进入思考模式")
                                    if not content: continue  # 如果只剩空字符则跳过

                                if "</think>" in content:
                                    is_in_thinking_tag = False
                                    content = content.replace("</think>", "")
                                    logging.info("检测到</think>标签，退出思考模式")
                                    if not content: continue

                                # 根据当前状态分发 Token 类型
                                if is_in_thinking_tag:
                                    logging.info(f"思考内容: {content[:50]}...")
                                    yield {"type": "reasoning_token", "content": content}
                                else:
                                    yield {"type": "token", "content": content}

                            elif "type" in data_json and data_json["type"] == "error":
                                yield {"type": "token", "content": f"Error: {data_json.get('content')}"}
                        except Exception as e:
                            logging.error(f"Stream parse error: {e}, line: {line_text[:100]}")
                            continue
    except Exception as e:
        logging.error(f"Stream connection error: {e}")
        yield {"type": "token", "content": f"Error during streaming: {str(e)}"}


async def stream_next_question_generator(question: str, answer: str, model_type: str) -> AsyncGenerator[str, None]:
    """
    负责调用大模型并逐字yield token (下一步问题预测)
    """
    prompt = (
        f"用户提出的问题：{question}\n大模型返回的答案：{answer}\n"
        "请预测下一步用户可能会问的n个问题（最多4个），格式如下：\n"
        "{'0':'question_1','1':'question_2', ...}\n"
        "请只返回JSON字符串，不要包含多余文字。\n"
    )
    payload = {"message": prompt, "model": model_type}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(LLM_URL, json=payload) as response:
                if response.status != 200:
                    yield ""  # 错误时静默或返回空
                    return

                async for line in response.content:
                    line_text = line.decode('utf-8').strip()
                    if line_text.startswith("data: "):
                        data_str = line_text[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data_json = json.loads(data_str)
                            if "choices" in data_json:
                                content = data_json["choices"][0]["delta"].get("content", "")
                                if content:
                                    yield content
                        except Exception:
                            continue
    except Exception as e:
        logging.error(f"Error streaming next questions: {e}")
        yield ""


async def distribute_tasks_generator(
        user_config: dict,
        user_question: str,
        ocr_text: Optional[str] = None
):
    """
    调度中心生成器 - 并行调用、顺序发送版
    集成了 RAG 检索、历史记忆、意图识别及多业务“先汇总、后逐个发送”处理[cite: 3]
    """

    # --- 1. 初始化公共上下文与历史记录 ---
    conv_id = user_config.get("conv_id") or user_config.get("chat_id")
    history_messages = []
    user_only_history = ""

    if conv_id:
        logging.info(f"正在获取会话 {conv_id} 的历史记录...")
        _, history_messages, user_only_history = await get_chat_history(str(conv_id))

    # 构造组合问题[cite: 3]
    combined_question = user_question
    if ocr_text:
        combined_question = (
            f"用户的文字提问: 在通信领域里：{user_question}\n用户的图片提问里的文字: {ocr_text}\n"
            f"请结合上述两个信息，生成一个连贯的问题。"
        )

    model = user_config.get('model', 'DeepSeek')
    is_reasoning = user_config.get('is_reasoning', False)

    # 所有对外SSE事件进入同一队列，按实际完成时间发送，避免业务Section被文字流长期阻塞。
    event_queue = asyncio.Queue()
    request_id = str(uuid.uuid4())
    request_started = time.perf_counter()

    async def emit(payload: Dict[str, Any]):
        await event_queue.put(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n")

    async def trace(stage: str, status: str, started: float = None, **metadata):
        now = time.perf_counter()
        payload: Dict[str, Any] = {
            "type": "trace", "request_id": request_id, "stage": stage, "status": status,
            "elapsed_ms": round((now - request_started) * 1000, 3),
        }
        if started is not None:
            payload["duration_ms"] = round((now - started) * 1000, 3)
        if metadata:
            payload["metadata"] = metadata
        await emit(payload)

    # --- Worker A: 快速回答通道 (大模型概念回答) ---
    async def answer_worker():
        rag_started = time.perf_counter()
        try:
            # RAG 检索逻辑[cite: 3]
            rag_chunks_text = ""
            retrieved_count = 0
            await trace("rag", "start")
            if ragflow_client and target_dataset_ids_for_ragflow:
                try:
                    url = f"{RAGFLOW_SDK_BASE_URL}/api/v1/retrieval"
                    headers = {"Authorization": f"Bearer {RAGFLOW_API_KEY}"}
                    payload = {
                        "dataset_ids": target_dataset_ids_for_ragflow,
                        "question": combined_question,
                        "page": 1,
                        "page_size": RAGFLOW_CANDIDATE_K,
                        "similarity_threshold": RAGFLOW_SIMILARITY_THRESHOLD,
                        "vector_similarity_weight": RAGFLOW_VECTOR_WEIGHT,
                        "top_k": RAGFLOW_EMBEDDING_TOP_K,
                    }
                    async with aiohttp.ClientSession() as session:
                        async with session.post(url, headers=headers, json=payload, timeout=10) as resp:
                            if resp.status == 200:
                                rag_response = await resp.json()
                                logging.info("RAGFlow API 响应成功")

                                # 核心修正：获取嵌套的 chunks 列表
                                data_obj = rag_response.get('data', {})
                                if isinstance(data_obj, dict):
                                    retrieved_items = data_obj.get('chunks', [])
                                else:
                                    retrieved_items = data_obj if isinstance(data_obj, list) else []

                                # ==========================================
                                # 【关键修复 1】：不管 API 返回多少条，我们强制只取前 4 条最相关的！
                                # 防止海量数据撑爆大模型的上下文和思考空间
                                # ==========================================
                                if len(retrieved_items) > RAGFLOW_CONTEXT_K:
                                    retrieved_items = retrieved_items[:RAGFLOW_CONTEXT_K]
                                retrieved_count = len(retrieved_items)

                                if len(retrieved_items) > 0:
                                    logging.info(f"✅ 知识库调用成功！实际采纳 {len(retrieved_items)} 条片段")
                                    chunks_builder = ["\n\n以下是知识库中检索到的相关参考信息："]
                                    for i, item in enumerate(retrieved_items):
                                        content = item.get('content_with_weight') or item.get('content') or ""
                                        if content:
                                            score = item.get('similarity', 0)
                                            chunks_builder.append(f"参考片段 {i + 1} (匹配度:{score:.4f}): {content}")
                                    rag_chunks_text = "\n".join(chunks_builder)
                                else:
                                    logging.warning("⚠️ 检索请求成功，但 data.chunks 中没有内容。")
                except Exception as e:
                    logging.error(f"RAG 检索逻辑执行出错: {e}")
                    await trace("rag", "error", rag_started, error_type=type(e).__name__)

            await trace("rag", "end", rag_started, chunks=retrieved_count)

            # 流式回答生成[cite: 3]
            full_answer_accumulator = ""
            llm_started = time.perf_counter()
            first_answer_emitted = False
            await trace("llm_answer", "start")
            async for item in stream_concept_answer_generator(
                    combined_question, model, rag_chunks_text, history_messages, is_reasoning
            ):
                if item["type"] == "token":
                    full_answer_accumulator += item["content"]
                    if item.get("content") and not first_answer_emitted:
                        first_answer_emitted = True
                        await trace("llm_answer", "first_token", llm_started)
                await emit(item)

            await trace("llm_answer", "end", llm_started,
                        answer_chars=len(full_answer_accumulator))
            # 主答案完成。前端收到此事件后独立调用 /next_questions，不再阻塞主Workflow。
            await emit({
                "type": "answer_complete", "request_id": request_id,
                "question": combined_question, "model": model,
            })

        except Exception as e:
            logging.error(f"Answer worker error: {e}")
            await trace("llm_answer", "error", error_type=type(e).__name__)
            await emit({'type': 'token', 'content': f' Error: {str(e)}'})
        finally:
            await event_queue.put({"internal": "worker_done", "worker": "answer"})

    # --- Worker B: 意图识别与业务顺序分发 ---
    async def intent_worker():
        intent_started = time.perf_counter()
        try:
            # B-1. 意图识别：确定哪些业务需要执行
            business_types = copy.deepcopy(business_types_template)
            await trace("intent", "start")
            try:
                identified_config = await auto_select_business_type(combined_question, model)
                business_types.update(identified_config)
            except Exception as e:
                logging.error(f"Intent recognition failed: {e}")
                business_types['问题内容'] = combined_question

            business_types["大模型概念回答"] = 1
            final_question = business_types.get('问题内容', combined_question)

            # 旧LLM路由模式保留硬规则兜底；规则路由已经进行更严格的实验意图判断，
            # 避免“硬件上是否节省”等概念题误开实验面板。
            hardware_force_question = final_question or combined_question or user_question
            if (INTENT_ROUTER_MODE == "llm"
                    and should_force_online_hardware_experiment(hardware_force_question)):
                if business_types.get("通信原理课堂在线硬件实验") != 1:
                    logging.info(
                        "[硬件实验强制启动] 检测到实验意图，覆盖 LLM 意图识别结果: "
                        f"question='{hardware_force_question}', "
                        f"old_online_hardware={business_types.get('通信原理课堂在线硬件实验')}"
                    )
                business_types["通信原理课堂在线硬件实验"] = 1

            selected_keys = [
                key for key, enabled in business_types.items()
                if key != "问题内容" and enabled == 1
            ]
            await trace("intent", "end", intent_started, selected=selected_keys,
                        router_mode=INTENT_ROUTER_MODE)

            # 尽早发送 UI 布局配置（告知前端哪些板块该转圈或显示）[cite: 5]
            await emit({'type': 'config', 'data': business_types})

            # B-2. 构建并行任务列表
            tasks_to_run = []
            keys_to_run = []

            # 业务映射表：业务名称 -> 执行函数[cite: 5]
            mapping = {
                "通信原理题目库": lambda: asyncio.to_thread(fetch_question_bank_sync, final_question),
                "通信原理课堂教师答疑记录": lambda: asyncio.to_thread(fetch_qa_record_sync, final_question),
                "通信原理课堂交互式案例": lambda: asyncio.to_thread(fetch_interactive_stories_sync, final_question),
                "通信原理课堂在线硬件实验": lambda: fetch_hardware_experiments_async(
                    final_question,
                    model,
                    history_messages=history_messages,
                    conv_id=str(conv_id) if conv_id else ""
                ),
                "其他课程实验案例": lambda: fetch_other_course_experiments(final_question, model),
                "思政教育案例": lambda: fetch_ideology_cases(final_question, model)  # <--- 新增这一行
            }

            # 遍历模板，筛选出本次需要运行的业务
            for key, func in mapping.items():
                if business_types.get(key) == 1:
                    keys_to_run.append(key)
                    tasks_to_run.append(func())

            async def run_business(key: str, awaitable):
                stage_started = time.perf_counter()
                await trace("business", "start", key=key)
                try:
                    result_data = await awaitable
                    await emit({'type': 'section', 'key': key, 'data': result_data})
                    await trace("business", "end", stage_started, key=key)
                except Exception as exc:
                    logging.error(f"Task {key} failed: {exc}")
                    await emit({
                        'type': 'section', 'key': key,
                        'data': {"content": f"服务暂时不可用 ({key})", "list": []}
                    })
                    await trace("business", "error", stage_started,
                                key=key, error_type=type(exc).__name__)

            # 每个业务完成后立即emit Section；gather只负责等待所有包装任务收尾，不再批量拼接结果。
            if tasks_to_run:
                await asyncio.gather(*[
                    run_business(key, awaitable)
                    for key, awaitable in zip(keys_to_run, tasks_to_run)
                ])

        except Exception as e:
            logging.error(f"Intent worker error: {e}")
            await trace("intent", "error", intent_started, error_type=type(e).__name__)
        finally:
            await event_queue.put({"internal": "worker_done", "worker": "intent"})

    # --- 2. 启动并行 Worker ---
    task_a = asyncio.create_task(answer_worker())
    task_b = asyncio.create_task(intent_worker())

    # --- 3. 按事件实际到达顺序发送；Section不会再被答案或推荐追问队列阻塞。 ---
    completed_workers = set()
    while len(completed_workers) < 2:
        item = await event_queue.get()
        if isinstance(item, dict) and item.get("internal") == "worker_done":
            completed_workers.add(item["worker"])
            continue
        yield item

    await trace("workflow", "end", request_started)
    while not event_queue.empty():
        item = await event_queue.get()
        if isinstance(item, str):
            yield item
    yield "data: [DONE]\n\n"
    logging.info("=== distribute_tasks_generator 任务分发执行完毕 ===")


async def next_questions_response_generator(request: NextQuestionRequest):
    """独立生成推荐追问，不占用主Agent流的完成时间。"""
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    yield f"data: {json.dumps({'type': 'next_questions_start', 'request_id': request_id}, ensure_ascii=False)}\n\n"
    yield f"data: {json.dumps({'type': 'trace', 'request_id': request_id, 'stage': 'next_questions', 'status': 'start', 'elapsed_ms': 0.0}, ensure_ascii=False)}\n\n"
    try:
        async for token in stream_next_question_generator(request.question, request.answer, request.model):
            if token:
                yield f"data: {json.dumps({'type': 'next_question_token', 'content': token}, ensure_ascii=False)}\n\n"
        duration_ms = round((time.perf_counter() - started) * 1000, 3)
        yield f"data: {json.dumps({'type': 'trace', 'request_id': request_id, 'stage': 'next_questions', 'status': 'end', 'elapsed_ms': duration_ms, 'duration_ms': duration_ms}, ensure_ascii=False)}\n\n"
    except Exception as exc:
        duration_ms = round((time.perf_counter() - started) * 1000, 3)
        logging.error(f"Next-question worker failed: {exc}")
        yield f"data: {json.dumps({'type': 'trace', 'request_id': request_id, 'stage': 'next_questions', 'status': 'error', 'elapsed_ms': duration_ms, 'duration_ms': duration_ms, 'metadata': {'error_type': type(exc).__name__}}, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/next_questions")
async def next_questions(request: NextQuestionRequest):
    return StreamingResponse(next_questions_response_generator(request), media_type="text/event-stream")

@app.post("/tasks_distribute")
async def tasks_distribute(
        config: str = Form(...), entrance: str = Form(...), locale: str = Form(...),
        image: Optional[UploadFile] = File(None),
):
    try:
        # 解析配置
        message_from_form_str = config
        str_data = message_from_form_str.replace("'", '"')
        message_config = json.loads(str_data)
        logging.info(f"Received config: {message_config}")  # 新增

        # 处理图片 OCR
        ocr_text = None
        if image is not None:
            temp_file_path = await save_image_locally(image)
            ocr_text = await get_ocr_text_async_wrapper(temp_file_path)
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        user_original_question = message_config.get('content', '')

        # 返回 StreamingResponse，启用 SSE
        return StreamingResponse(
            distribute_tasks_generator(message_config, user_original_question, ocr_text),
            media_type="text/event-stream"
        )

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Config JSON Error: {str(e)}")
    except Exception as e:
        logging.error(f"Main error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    # 启动服务
    uvicorn.run(app, host="0.0.0.0", port=9450)
