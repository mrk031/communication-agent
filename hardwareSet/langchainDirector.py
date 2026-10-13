# langchainTestServer.py
# -*- coding: utf-8 -*-
from fastapi import FastAPI, Request
import langchain
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from pydantic import BaseModel
from typing import List
import logging
import os
# 替换为langchain_classic的导入路径
from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains import LLMChain, RetrievalQA
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.agents import initialize_agent, AgentType
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from TCPClientHardware import HardwareClient
from output_schama import HardWareParameterSchema
# 保留langchain_core的Tool导入，删除langchain.agents的重复Tool导入
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import functools
import math

# 示波器测量的波形类型
sbq_boxing_type = '''
                '基带时钟波形':30,
                '基带信号波形':31,
                '差分编码波形':73,
                '载波频率波形':57,
                '相移180度的载波频率波形':56,
                '调制输出波形':62,
                '解调输入波形':100,
                'I路解调输出波形':117,
                'I路解调通过低通滤波器输出波形':119,
                'Q路解调输出波形':116,
                'Q路解调通过低通滤波器输出波形':118,
                '本地载波波形':115,
                '抽样判决门限':105,
                '抽样后信号的输出波形':120,
                '解调输出的波形':122,
                '恢复时钟的波形':123,
                '噪声波形':49
                '''
# 示波器通道数映射关系
sbq_tongdaoshu = '''
                1:'0x8011',
                2:'0x8012',
                3:'0x8013',
                4:'0x8014'
                '''
# 调制解调开关的操作类型映射关系
recode_io = '''
      打开调制解调开关:0,
      闭合调制解调开关:1
         '''
# 基带发送信号码长度提取的映射关系
jd_send_xinhao_map = '''
     基带发送信号码取值的映射关系如下。冒号前面是提取值，冒号后面是基带送信号码类型描述。
     0:基带发送一个全0码,
     1:基带发送一个全1码,
     x:基带发送一个长度为x长度的m序列,
     27:基带发送一个0110……（长度小于8）
     26:基带发送一个0110……（长度8-16）
     31:基带发送一个0110……（长度17-32）
     '''


# 只获取列表,大列表套小列表，需要读出来再组合
def get_final_list(llm_out_text):
    spl = llm_out_text.replace('"', '').replace("'", '').split('[')[-1].split(']')[0]
    final_list = [x for x in spl.split(',')]
    return final_list


llm = ChatOpenAI(
    temperature=0.5,  # 设置随机性
    max_tokens=None,
    timeout=None,
    max_retries=2,
    model=os.getenv("HARDWARE_LLM_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("HARDWARE_LLM_API_KEY"),
    base_url=os.getenv("HARDWARE_LLM_BASE_URL", "https://api.openai.com/v1")
)

HARDWARE_TCP_HOST = os.getenv("HARDWARE_TCP_HOST", "127.0.0.1")
HARDWARE_TCP_PORT = int(os.getenv("HARDWARE_TCP_PORT", "9001"))

# 设置日志记录
logging.basicConfig(level=logging.DEBUG)

# 将所有工具和模型组合成一个代理
tools = []
llm_with_tools = llm.bind_tools(tools)
structured_output_llm = llm.with_structured_output(schema=HardWareParameterSchema)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f'''你是一个信息提取的高手。你能从用户的输入的文本:中准确提取出我们需要的指定的参数值。并将参数值按照顺序放入列表中输出。\
             你需要提取的参数有10个，名称为:抽样判决、环路滤波、VCO、基带时钟、载波频率、测量输出路型、基带发送信号码、噪声源系数、示波器的通道数和测量波形类型的参数组合值、调制解调开关操作类型。提取出来的10个参数值按照[抽样判决,环路滤波,VCO,基带时钟,载波频率,测量输出路型,基带发送信号码,噪声源系数,示波器的通道数和测量波形类型的参数组合值,调制解调开关操作类型]的顺序放入列表中，再输出。\
             如果文本中没有提到某个参数，则列表中这个参数的取值设为-100.
             需要注意的地方：
             1、基带发送信号码特殊描述提取规则说明：
               如果基带发送信号码的相关描述为：'基带发送一个长度为x长度的m序列'时，提取的基带发送信号码参数值为x。
               如果基带发送信号码的相关描述为:'基带发送一个全0码',则提取0。
               如果基带发送信号码的相关描述为：'基带发送一个全1码',则提取1。
               如果基带发送信号码的相关描述不是以上3种情况，则正常提取对应的参数内容即可。
             2、示波器的通道数的参数值提取按照{sbq_tongdaoshu}来进行映射。例如通道数为1，则提取的参数值为'0x8011'.
             3、示波器测量的波形类型需要根据相关描述准确映射为对应的数字。文字描述与对应数据映射关系为{sbq_boxing_type}。要特别注意其中映射值为
             116和118的描述区别。
             4、示波器的通道数和测量波形类型的参数组合值参数提取规则说明：示波器的通道数和示波器测量的波形类型是同时成对出现的，在描述中可能会多次出现。需要将将示波器的通道数和示波器测量的波形类型2个参数的映射后的提取值进行组合作为示波器的通道数和测量波形类型的参数组合值。
               将每个描述中的通道数和波形类型组合均按对形成组合值，作为示波器的通道数和测量波形类型的参数组合值的提取值，组合形式为:'通道数a@波形类型a|通道数b@波形类型b ....'。描述中有几对组合，就提取几对。如果没有相关的描述，则提取值为'-100:-100'
               比如描述为:'用通道1测量基带时钟波形，通道4测量载波频率波形',则提取组合值为：'0x8011@30|0x8014:57'。
               再比如描述为：'用通道1测量基带时钟波形',则提取的组合值'0x8011@30'。

             5、调制解调开关操作类型的参数也根据相关描述准确映射为对应的数字.其文字描述与对应数据映射关系为{recode_io}。

             6、输出要带上单位。
             例如1：输入文本是：dpsk实验 设置载波频率为39khz,VCO为47khz,环路滤波器为99hz,抽样判决为77hz,基带时钟为16K,测量I路相干输出,基带发送一个全1码,发送一个噪声系数为100,用示波器通道3测量Q路解调输出波形,打开调制解调开关。
               则输出：[77,99hz,47khz,16K,39khz,I,1,100,'0x8013@116',0]
             例如2:输入文本是：dpsk实验 设置载波频率为39khz,VCO为47khz,环路滤波器为99hz,抽样判决为77hz,基带时钟为16K,测量I路相干输出,基带发送一个0111001信号码,发送一个噪声系数为100,用示波器通道3测量恢复时钟的波形,通道2测量差分编码波形,打开调制解调开关。
               则输出：[77,99hz,47khz,16K,39khz,I,'0111001',100,'0x8013@123|0x8012@73',0]
             '''
        ),
        ("user", "{input_text}"),
    ]
)
llm_with_tools_chain = prompt | llm_with_tools
structured_output_llm_chain = prompt | structured_output_llm


# 只获取最终的列表
def get_final_list(llm_out_text):
    spl = llm_out_text.replace('"', '').replace("'", '').split('[')[-1].split(']')[0]
    final_list = [x for x in spl.split(',')]
    return final_list


app = FastAPI()


class InstructionRequest(BaseModel):
    instruction_to_send: str


@app.post("/GetArgs")
async def get_message_from_words(request: Request):
    # 通过用户语言获得对应的配置
    # 用户的数据格式：{"message":"xxx","model":"XH"}   # 模型选择：XH、Llama、DeepSeek
    user_message = await request.json()
    if 'message' not in user_message:
        return {"error": "Message content is required."}
    message = user_message['message']
    logging.info("message:" + message)
    print('接收的文本为: ', message)
    query = {"input": f'{message}'}
    ai_msg = llm_with_tools_chain.invoke(query['input'])
    logging.debug("ai_back:" + ai_msg.content)

    ret_content = ai_msg.content
    final_list = get_final_list(ret_content)
    print('大模型初始输出列表', final_list)

    logging.debug("ai_back:" + ai_msg.content)

    # 抽样判决,环路滤波,VCO,基带时钟,载波频率、IO切换,
    cypj, hllb, vco, clock, carrier_f, ioSwitch = final_list[0], final_list[1], final_list[2], final_list[
        3].strip().upper(), final_list[4], final_list[5].strip().upper()
    # 基带发送信号码、噪声源系数、示波器通道数_示波器测量的波形类型、调制解调开关操作类型
    jd_send_xinhao_map_val, zsy, sbq_tongdaoshu_boxing_type, recode_io_val = final_list[6].strip(), int(final_list[7]), \
    final_list[8].strip(), int(final_list[9])
    sbq_tongdaoshu_boxing_type_map = sbq_tongdaoshu_boxing_type.replace('|', ',').replace('@', ',').split(',')
    sbq_tongdaoshu_boxing_type_map = sbq_tongdaoshu_boxing_type_map if sbq_tongdaoshu_boxing_type_map != ['-100',
                                                                                                          '-100'] else [
        '0x8011', 116, '0x8012', 118]

    if ioSwitch == 'I':
        ioSwitch = 1
    elif ioSwitch == 'Q':
        ioSwitch = 2
    else:
        ioSwitch = 2

    clock_map = {
        '256': 254,
        '512': 255,
        '1K': 0,
        '2K': 1,
        '4K': 2,
        '8K': 3,
        '16K': 4,
        '32K': 5,
        '64K': 6,
        '128K': 7,
        '256K': 8,
        '512K': 9,
        '1024K': 10,
        '2048K': 11,
        '4096K': 12
    }
    clock_map = clock_map.get(clock, 2)

    # 判断是否包括0，1之外的字符
    def contains_only_01(s):
        return all(c in '01' for c in s)

    # 最终映射取值以及十进制的值
    def get_jd_send_xinhao_map_fial(jd_send_xinhao_map_val):
        print('jd_send_xinhao_map', jd_send_xinhao_map_val)

        BIT8 = 128
        BIT16 = 128
        BIT32 = 128

        # 先清理空白字符
        val_str = jd_send_xinhao_map_val.strip()

        # 1. 优先处理默认值 -100 的情况，直接跳过计算
        if val_str == '-100':
            jd_send_xinhao_final = -100
            jd_send_xinhao_final_10 = -100

        # 2. 处理单字符 0 或 1
        elif len(val_str) == 1 and val_str in ['0', '1']:
            jd_send_xinhao_final = int(val_str)  # 基带发送一个全0码或全1码
            jd_send_xinhao_final_10 = -100  # 十进制 不取值

        # 3. 处理 m序列长度 (例如 "7" -> log2(7)? 或者是 "x" 提取的数值)
        # 注意：这里加了 val_int > 0 的判断，防止算 log2 报错
        elif not contains_only_01(val_str):  # 包括01之外的字符
            try:
                val_int = int(val_str)
                if val_int > 0:
                    jd_send_xinhao_final = int(math.log2(val_int))
                else:
                    jd_send_xinhao_final = -100
            except ValueError:
                # 防止 int() 转换失败
                jd_send_xinhao_final = -100
            jd_send_xinhao_final_10 = -100  # 十进制 不取值

        # 4. 处理纯二进制字符串 (例如 "0110")
        elif contains_only_01(val_str):  # 不包括01之外的数据，纯二进制
            jd_send_xinhao_final_10 = int(val_str, 2)  # 转后的十进制值
            lens = len(val_str)
            if lens < 8:
                jd_send_xinhao_final = 27
                BIT8 = jd_send_xinhao_final_10
            elif 8 < lens <= 16:
                jd_send_xinhao_final = 26
                BIT16 = jd_send_xinhao_final_10
            elif 17 <= lens <= 32:
                jd_send_xinhao_final = 31
                BIT32 = jd_send_xinhao_final_10
            else:
                jd_send_xinhao_final = -100
        else:
            jd_send_xinhao_final = -100
            jd_send_xinhao_final_10 = -100

        return jd_send_xinhao_final, jd_send_xinhao_final_10, BIT8, BIT16, BIT32

    # 基带发送码最终映射值及其十进制转换值
    jd_send_xinhao_final, jd_send_xinhao_final_10, BIT8, BIT16, BIT32 = get_jd_send_xinhao_map_fial(
        jd_send_xinhao_map_val)
    jd_send_xinhao_final = jd_send_xinhao_final if int(jd_send_xinhao_final) != -100 else 2

    cypj = cypj if int(cypj) != -100 else 150
    hllb = hllb if hllb != -100 else 200
    vco = vco if vco != -100 else 256
    carrier_f = carrier_f if carrier_f != -100 else 155
    zsy = zsy if zsy != -100 else 100
    recode_io_val = recode_io_val if recode_io_val != -100 else 1

    # khz与hz 转换，提取以khz为单位的结果
    def khz_map(ori_data):
        ret = ori_data
        if 'khz' in ori_data:
            tmp = ori_data.replace('khz', '')
            ret = tmp.strip()
            return ret
        elif 'hz' in ori_data:
            tmp = int(ori_data.replace('hz', ''))
            ret = tmp / 1000
        else:
            ret = ori_data
        return ret

    carrier_f = khz_map(carrier_f)
    hllb = khz_map(hllb)
    vco = khz_map(vco)

    if float(carrier_f) < 0 or float(carrier_f) > 256:
        print(f'输入载波频率为 {carrier_f}，超出范围，请重新输入参数。当前已经转为默认值155。')
        carrier_f = 155

    # 要发送的消息
    message_to_send = {
        "MC": "DPSK",  # 当前实验
        "OUT": jd_send_xinhao_final,  # 基带输出
        "CLOCK": clock_map,  # 基带时钟
        "BIT8": BIT8,  # 8Bit拨码开关
        "BIT16": BIT16,  # 16Bit
        "BIT32": BIT32,  # 32Bit
        "CarrierF": carrier_f,  # 载波频率
        "VCOF": vco,  # VCO
        "IOSwitch": ioSwitch,  # IO切换
        "PhaseC": 1,  # 相位控制
        "CYPJ": cypj,  # 抽样判决
        "HLLB": hllb,  # 环路滤波
        "ZSY": zsy,  # 噪声源
        "LINK": sbq_tongdaoshu_boxing_type_map,  # 连线
        "recodeIO": recode_io_val  # 调制解调开关
    }

    print(cypj, hllb, vco, clock_map, carrier_f, ioSwitch, jd_send_xinhao_final, jd_send_xinhao_final_10, BIT8, BIT16,
          BIT32, zsy, sbq_tongdaoshu_boxing_type_map, recode_io_val)
    print('message_to_send: ', message_to_send)

    # 融合代码，向硬件发送参数
    # 硬件服务端地址和端口
    server_host = HARDWARE_TCP_HOST
    server_port = HARDWARE_TCP_PORT
    # 调用函数发送消息
    hardware_client = HardwareClient(server_host, server_port)
    hardware_client.send_string_to_server(message_to_send)
    return {"HardWareArgs": message_to_send}


@app.post("/SetArgs")
async def set_args_to_hardware(request: Request):
    # 根据获得的配置参数取配置数据
    message_to_send = await request.json()
    # 硬件服务端地址和端口
    server_host = HARDWARE_TCP_HOST
    server_port = HARDWARE_TCP_PORT
    # 调用函数发送消息
    hardware_client = HardwareClient(server_host, server_port)
    hardware_client.send_string_to_server(message_to_send)
    return "okk"


# 测试接口调用
def test_post():
    import requests
    url = "http://127.0.0.1:9455/GetArgs"
    data = {"name": "Alice",
            "message": "dpsk实验，VCO为78khz,载波频率为345khz,环路滤波是22hz，基带时钟为8K, 测量Q路相干输出，基带发送一个长度为7的m序列,发送一个噪声系数为90,用示波器通道4测量Q路解调输出波形,通道1测量基带时钟波形，闭合调制解调开关。"}
    response = requests.post(url, json=data)


if __name__ == "__main__":
    import uvicorn
    import nest_asyncio  # 允许嵌套事件循环

    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=9455)
