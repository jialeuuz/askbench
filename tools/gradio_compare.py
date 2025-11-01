import gradio as gr
import requests
import json
from typing import Generator, List, Dict

def call_llm_api(api_url: str, messages: List[Dict], api_key: str = "EMPTY") -> Generator[str, None, None]:
    """
    调用OpenAI格式的LLM API（支持流式输出）
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": "default",
        "messages": messages,
        "stream": True,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=data, stream=True, timeout=60)
        response.raise_for_status()
        
        full_response = ""
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    line = line[6:]
                    if line.strip() == '[DONE]':
                        break
                    try:
                        json_data = json.loads(line)
                        if 'choices' in json_data and len(json_data['choices']) > 0:
                            delta = json_data['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                full_response += content
                                yield full_response
                    except json.JSONDecodeError:
                        continue
        
        if not full_response:
            yield "⚠️ 未收到有效响应"
            
    except requests.exceptions.RequestException as e:
        yield f"❌ 请求错误: {str(e)}"
    except Exception as e:
        yield f"❌ 发生错误: {str(e)}"

def call_llm_api_non_stream(api_url: str, messages: List[Dict], api_key: str = "EMPTY") -> str:
    """
    调用OpenAI格式的LLM API（非流式输出）
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": "default",
        "messages": messages,
        "stream": False,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            return "⚠️ 未收到有效响应"
            
    except requests.exceptions.RequestException as e:
        return f"❌ 请求错误: {str(e)}"
    except Exception as e:
        return f"❌ 发生错误: {str(e)}"

def format_chat_history(history: List[List]) -> str:
    """
    格式化聊天历史显示
    """
    if not history:
        return ""
    
    formatted = ""
    for i, (user_msg, bot_msg) in enumerate(history, 1):
        formatted += f"**[第{i}轮对话]**\n\n"
        formatted += f"👤 **用户**: {user_msg}\n\n"
        if bot_msg:
            formatted += f"🤖 **助手**: {bot_msg}\n\n"
        formatted += "---\n\n"
    return formatted

def build_messages(history: List[List], current_question: str) -> List[Dict]:
    """
    构建完整的消息列表（包含历史对话）
    """
    messages = []
    
    # 添加历史对话
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
    
    # 添加当前问题
    if current_question:
        messages.append({"role": "user", "content": current_question})
    
    return messages

def compare_models(question: str, api1_url: str, api2_url: str, use_stream: bool, 
                  history1: List[List], history2: List[List]):
    """
    同时调用两个模型API并返回结果（带上下文）
    """
    if not question.strip():
        return history1, history1, "", history2, history2, ""
    
    # 构建消息列表
    messages1 = build_messages(history1, question)
    messages2 = build_messages(history2, question)
    
    if use_stream:
        # 流式输出
        gen1 = call_llm_api(api1_url, messages1)
        gen2 = call_llm_api(api2_url, messages2)
        
        response1 = ""
        response2 = ""
        done1, done2 = False, False
        
        # 先添加用户问题到历史
        new_history1 = history1 + [[question, None]]
        new_history2 = history2 + [[question, None]]
        
        while not (done1 and done2):
            try:
                if not done1:
                    response1 = next(gen1)
            except StopIteration:
                done1 = True
            
            try:
                if not done2:
                    response2 = next(gen2)
            except StopIteration:
                done2 = True
            
            # 更新历史记录
            temp_history1 = history1 + [[question, response1]]
            temp_history2 = history2 + [[question, response2]]
            
            yield (temp_history1, format_chat_history(temp_history1), "", 
                   temp_history2, format_chat_history(temp_history2), "")
        
        # 最终结果
        final_history1 = history1 + [[question, response1]]
        final_history2 = history2 + [[question, response2]]
        
        return (final_history1, format_chat_history(final_history1), "",
                final_history2, format_chat_history(final_history2), "")
    else:
        # 非流式输出
        response1 = call_llm_api_non_stream(api1_url, messages1)
        response2 = call_llm_api_non_stream(api2_url, messages2)
        
        new_history1 = history1 + [[question, response1]]
        new_history2 = history2 + [[question, response2]]
        
        return (new_history1, format_chat_history(new_history1), "",
                new_history2, format_chat_history(new_history2), "")

def clear_history():
    """
    清空对话历史，开始新对话
    """
    return [], "", "", [], "", ""

# 创建Gradio界面
with gr.Blocks(title="LLM模型对比工具", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🤖 LLM模型对比工具
        同时测试两个模型的回复效果，支持多轮对话，方便对比分析
        """
    )
    
    # 状态变量：存储两个模型的对话历史
    chat_history1 = gr.State([])
    chat_history2 = gr.State([])
    
    with gr.Row():
        with gr.Column():
            api1_input = gr.Textbox(
                label="模型1 API地址",
                value="http://10.80.13.48:8012/v1/chat/completions",
                placeholder="输入第一个API的完整URL"
            )
        with gr.Column():
            api2_input = gr.Textbox(
                label="模型2 API地址",
                value="http://10.80.13.48:8013/v1/chat/completions",
                placeholder="输入第二个API的完整URL"
            )
    
    with gr.Row():
        question_input = gr.Textbox(
            label="输入问题",
            placeholder="在这里输入你想问的问题...",
            lines=3,
            scale=4
        )
        with gr.Column(scale=1):
            stream_checkbox = gr.Checkbox(label="启用流式输出", value=True)
            with gr.Row():
                submit_btn = gr.Button("🚀 发送", variant="primary", size="lg")
                clear_btn = gr.Button("🔄 新对话", variant="secondary", size="lg")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📝 模型1对话历史")
            output1 = gr.Markdown(label="", height=500)
        with gr.Column():
            gr.Markdown("### 📝 模型2对话历史")
            output2 = gr.Markdown(label="", height=500)
    
    # 绑定提交事件
    submit_btn.click(
        fn=compare_models,
        inputs=[question_input, api1_input, api2_input, stream_checkbox, 
                chat_history1, chat_history2],
        outputs=[chat_history1, output1, question_input, 
                 chat_history2, output2, question_input]
    )
    
    # 支持回车提交
    question_input.submit(
        fn=compare_models,
        inputs=[question_input, api1_input, api2_input, stream_checkbox,
                chat_history1, chat_history2],
        outputs=[chat_history1, output1, question_input,
                 chat_history2, output2, question_input]
    )
    
    # 清空历史按钮
    clear_btn.click(
        fn=clear_history,
        inputs=[],
        outputs=[chat_history1, output1, question_input,
                 chat_history2, output2, question_input]
    )
    
    gr.Markdown(
        """
        ---
        ### 使用说明
        1. **配置API地址**: 确认两个模型的API地址是否正确
        2. **输入问题**: 在输入框中输入你想测试的问题
        3. **流式输出**: 可选择是否启用流式输出（实时看到生成过程）
        4. **发送问题**: 点击"🚀 发送"按钮或按回车键提交
        5. **多轮对话**: 继续输入问题即可进行多轮对话，模型会记住上下文
        6. **新对话**: 点击"🔄 新对话"按钮清空历史，开始新的对话
        7. **对比分析**: 左右对比两个模型的回复效果和上下文理解能力
        
        ### 功能特点
        - ✅ 支持多轮对话，自动保持上下文
        - ✅ 同时对比两个模型的表现
        - ✅ 支持流式和非流式输出
        - ✅ 完整的对话历史显示
        """
    )

# 启动应用
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7866,
        share=False
    )