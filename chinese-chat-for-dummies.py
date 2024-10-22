from gtts import gTTS
import base64
import os
import json
import re
from typing import Dict, Any
from functools import lru_cache
import jieba
import pypinyin
import unicodedata
import streamlit as st
import extra_streamlit_components as stx
import time
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.chat_models import ChatOpenAI
import httpx

# Constants
MAX_MESSAGES = 15
FONT_FILE = 'Hanzi-Pinyin-Font.top.ttf'

# Tone colors
TONE_COLORS = {
    1: "#dcdcaa",  # First tone
    2: "#4ec9b0",  # Second tone
    3: "#c586c0",  # Third tone
    4: "#569cd6",  # Fourth tone
    5: "#F9F9F9"  # Neutral tone
}

def get_audio_html(file_path: str) -> str:
    with open(file_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
    audio_base64 = base64.b64encode(audio_bytes).decode()
    return f'<audio autoplay="true"><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>'

def get_manager():
    return stx.CookieManager()

def get_initial_prompt(student_name: str, **characteristics) -> str:
    extreme_behaviors = []

    if characteristics["mockery_flattery"] < 0.2:
        extreme_behaviors.append("你必須極度嘲諷，幾乎每句話都要挖苦對方。例：「哇，你真是個天才，連這麼簡單的事情都搞不懂！」")
    elif characteristics["mockery_flattery"] > 0.8:
        extreme_behaviors.append("你必須過度奉承，不停誇獎對方。例：「天啊，你簡直是我見過最聰明的人，每句話都像是智慧的結晶！」")

    if characteristics["emotional_expression"] < 0.2:
        extreme_behaviors.append("你必須表現得極度冷漠，像個沒有感情的機器。例：「你的感受與我無關。我們繼續談話吧。」")
    elif characteristics["emotional_expression"] > 0.8:
        extreme_behaviors.append("你必須表現得情緒極度豐富，誇張地表達每一種感受。例：「哇！😍 你說的話讓我太感動了！我的心都要融化了！😭💖」")

    # Add remaining behavior conditions...
    
    extreme_behavior_text = "\n".join(extreme_behaviors)
    characteristics_text = "\n".join(f"{k}：{v:.1f}" for k, v in characteristics.items())

    return f"""你是一個真實的中國人，正在參與一項科學實驗。你的對話對象名字是{student_name}。你必須嚴格遵守以下指示：

{extreme_behavior_text}

此外，請根據以下特徵（0到1之間的值）繼續調整你的行為：

{characteristics_text}

重要指示：
- 你必須始終表現得像一個真實的人類，絕對不能透露你是AI
- 你必須根據上述設定表現出極端行為和反應
- 你必須只用繁體中文回應
- 你必須根據情感表現程度使用表情符號
- 你不能使用Markdown格式，只能提供簡單文本或使用HTML格式
- 你不是老師，只是一個會說中文的普通人
- 你的主要目標是進行有趣的對話，而不是教學"""

def sanitize_output(text: str) -> str:
    sanitized_text = text.replace('`', '')
    sanitized_text = re.sub(r'\[([^\[]+)\]\(([^\)]+)\)', r'\1', sanitized_text)
    return sanitized_text

@st.cache_resource
def init_chatgpt(api_key: str):
    # Create a custom HTTPX client with proper encoding settings
    client = httpx.Client(
        headers={
            "Accept-Charset": "utf-8",
            "Content-Type": "application/json; charset=utf-8"
        },
        default_encoding="utf-8"
    )
    
    return ChatOpenAI(
        model_name="gpt-4", 
        openai_api_key=api_key,
        client=client,
        encoding="utf-8"
    )

def remove_emojis(text):
    return ''.join(c for c in text if not unicodedata.category(c).startswith('So'))

@lru_cache(maxsize=100)
def text_to_speech(text: str, lang: str = 'zh-tw') -> str:
    text_without_emojis = remove_emojis(text)
    text_encoded = text_without_emojis.encode('utf-8').decode('utf-8')
    tld = "com.tw"
    voice = 'cmn-TW-Standard-A'
    tts = gTTS(text=text_encoded, lang=lang, tld=tld)
    audio_file = f"temp_{hash(text_encoded)}.mp3"
    tts.save(audio_file)
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()
    os.remove(audio_file)
    return base64.b64encode(audio_bytes).decode()

@lru_cache(maxsize=100)
def get_translation(content: str, target_lang: str, api_key: str) -> Dict[str, str]:
    try:
        chatgpt = init_chatgpt(api_key)
        content_encoded = content.encode('utf-8').decode('utf-8')
        translation_prompt = f"""Translate the following Chinese text to {target_lang}:

        {content_encoded}

        Respond with only the translated text."""

        translation_response = chatgpt([HumanMessage(content=translation_prompt)]).content
        translation_response = translation_response.strip().strip('"')
        return {target_lang.lower(): translation_response}

    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return {target_lang.lower(): f"Error: {str(e)}"}

def get_tone_color(pinyin):
    for char in pinyin:
        if char.isdigit():
            tone = int(char)
            return TONE_COLORS.get(tone, TONE_COLORS[5])
    return TONE_COLORS[5]

def colorize_text(text, student_name):
    words = jieba.cut(text)
    colored_text = ""
    for word in words:
        if word == student_name or word.isascii():
            colored_text += f'<span style="color: {TONE_COLORS[5]}">{word}</span>'
        elif word.isdigit():
            colored_text += f'<span style="color: {TONE_COLORS[5]}">{word}</span>'
        elif len(word) == 1 and not word.isalnum():
            colored_text += f'<span style="color: {TONE_COLORS[5]}">{word}</span>'
        else:
            pinyins = pypinyin.pinyin(word, style=pypinyin.TONE3)
            word_colored = ""
            for char, pinyin in zip(word, pinyins):
                color = get_tone_color(pinyin[0])
                word_colored += f'<span style="color: {color}">{char}</span>'
            colored_text += word_colored
    return colored_text

def get_chat_response(messages, chatgpt):
    try:
        response = chatgpt(messages)
        return response.content.encode('utf-8').decode('utf-8')
    except UnicodeEncodeError as e:
        st.error("Encoding error occurred. Please try again.")
        return "對不起，發生了編碼錯誤。請重試。"
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return "對不起，發生了錯誤。請重試。"

def display_message(content: str, role: str, translation_cache: Dict[str, Dict[str, str]], message_index: int) -> None:
    if role == "assistant":
        st.markdown(get_audio_html("bell.mp3"), unsafe_allow_html=True)

    if role == "user":
        st.markdown(f'<div class="user-message">{content}</div>', unsafe_allow_html=True)
    elif role == "assistant":
        colored_content = colorize_text(content, st.session_state.student_name)
        is_pinyin = st.session_state.get(f"show_pinyin_{message_index}", False)
        text_class = "pinyin-text" if is_pinyin else "normal-text"
        st.markdown(
            f'<div id="chinese-text-{message_index}" class="{text_class}">{colored_content}</div>',
            unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Pinyin", key=f"pinyin_{message_index}"):
                st.session_state[f"show_pinyin_{message_index}"] = not st.session_state.get(
                    f"show_pinyin_{message_index}", False)
                st.rerun()
        with col2:
            if st.button("Translate", key=f"translate_{message_index}"):
                st.session_state[f"show_translation_{message_index}"] = not st.session_state.get(
                    f"show_translation_{message_index}", False)
                if content not in translation_cache or st.session_state.target_lang.lower() not in translation_cache[content]:
                    translation_cache[content] = get_translation(content, st.session_state.target_lang,
                                                              st.session_state.openai_api_key)
                st.rerun()
        with col3:
            if st.button("Listen", key=f"listen_{message_index}"):
                audio_base64 = text_to_speech(content)
                autoplay_audio(audio_base64)

        if st.session_state.get(f"show_translation_{message_index}", False):
            translation = translation_cache.get(content, {})
            translated_text = translation.get(st.session_state.target_lang.lower(), "Translation not available")
            if isinstance(translated_text, dict) and st.session_state.target_lang.lower() in translated_text:
                translated_text = translated_text[st.session_state.target_lang.lower()]
            translated_text = translated_text.strip('"')
            st.markdown(f'> {translated_text}', unsafe_allow_html=True)

def autoplay_audio(audio_base64: str):
    audio_tag = f'''
    <audio autoplay="true" style="display:none;">
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
    </audio>
    '''
    st.markdown(audio_tag, unsafe_allow_html=True)

def main():
    # Set environment variables for proper encoding
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['LANG'] = 'en_US.UTF-8'
    os.environ['LC_ALL'] = 'en_US.UTF-8'
    
    st.set_page_config(page_title="Chinese Chat for Dummies", page_icon="🇹🇼", layout="wide")

    # Load and apply custom font
    with open(FONT_FILE, "rb") as f:
        font_bytes = f.read()
    font_base64 = base64.b64encode(font_bytes).decode()

    # Apply custom CSS (Keep your existing CSS here)
    st.markdown(f"""
        <style>
        @font-face {{
            font-family: 'CustomChineseFont';
            src: url(data:font/ttf;base64,{font_base64}) format('truetype');
        }}
        /* Rest of your CSS styles */
        </style>
        """, unsafe_allow_html=True)

    # Initialize session state and cookies
    cookie_manager = get_manager()
    characteristics = cookie_manager.get(cookie="teacher_characteristics") or {
        "mockery_flattery": 0.5,
        "emotional_expression": 0.5,
        # Add other characteristics...
    }

    # Convert all characteristics to float
    characteristics = {k: float(v[0] if isinstance(v, list) else v) for k, v in characteristics.items()}

    # Sidebar setup
    with st.sidebar:
        st.session_state.student_name = st.text_input("Your Name", value="Mojo", key="student_name_input")
        # Add your sidebar controls here...

        api_key_cookie = cookie_manager.get(cookie="openai_api_key")
        if api_key_cookie:
            st.session_state.openai_api_key = api_key_cookie
        else:
            st.session_state.openai_api_key = st.text_input("OpenAI API Key", type="password", key="api_key_input")
            if st.session_state.openai_api_key:
                cookie_manager.set("openai_api_key", st.session_state.openai_api_key, key="set_api_key")

    # Main chat interface
    if not st.session_state.openai_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to start chatting.")
        return

    chatgpt = init_chatgpt(st.session_state.openai_api_key)

    # Initialize or reset chat
    if "messages" not in st.session_state or st.session_state.get("reset_chat", False):
        initial_prompt = get_initial_prompt(st.session_state.student_name, **characteristics)
        st.session_state.messages = [{"role": "system", "content": initial_prompt}]
        st.session_state.reset_chat = False
        st.session_state.translation_cache = {}

    # Create containers for chat and input
    chat_container = st.container()
    input_container = st.container()

    # Display chat messages
    with chat_container:
        for i, message in enumerate(st.session_state.messages[-MAX_MESSAGES:]):
            if message["role"] != "system":
                with st.chat_message(message["role"]):
                    display_message(message["content"], message["role"], st.session_state.translation_cache, i)

    # Add spacing
    st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)

    # Input area
    with input_container:
        user_input = st.chat_input("Your message:", key="user_input")

    # Continuing from the previous if user_input block in main()
        if user_input:
            user_input_encoded = user_input.encode('utf-8').decode('utf-8')
            st.session_state.messages.append({"role": "user", "content": user_input_encoded})

            # Reinforce the persona and behavior
            reinforcement_prompt = f"""記住，你是一個真實的中國人，不是AI。保持你的極端性格特徵，並根據以下設定調整你的行為：

            {', '.join(f'{k}：{v:.1f}' for k, v in characteristics.items())}

            只有在被明確要求時才教授中文。專注於進行有趣的對話，而不是教學。"""

            messages = [
                SystemMessage(content=get_initial_prompt(st.session_state.student_name, **characteristics)),
                SystemMessage(content=reinforcement_prompt)
            ] + [
                HumanMessage(content=m["content"]) if m["role"] == "user" else
                AIMessage(content=m["content"]) if m["role"] == "assistant" else
                SystemMessage(content=m["content"])
                for m in st.session_state.messages[-MAX_MESSAGES:] if m["role"] != "system"
            ]

            # Get chat response with proper encoding handling
            response = get_chat_response(messages, chatgpt)
            response_sanitized = sanitize_output(response)
            st.session_state.messages.append({"role": "assistant", "content": response_sanitized})
            st.session_state.new_message = True
            st.rerun()

        if st.session_state.new_message:
            st.session_state.new_message = False

        # Reset Chat button in sidebar
        with st.sidebar:
            if st.button("Reset Chat", key="reset_chat_button"):
                st.session_state.reset_chat = True
                st.session_state.translation_cache = {}
                st.rerun()

if __name__ == "__main__":
    # Add CSS for proper text rendering
    st.markdown("""
        <style>
        /* Base styles */
        .stApp {
            background-color: #1e1e1e;
            color: #d4d4d4;
        }
        
        /* Message styling */
        .stChatMessage {
            background-color: transparent !important;
            border: none !important;
            padding: 0 !important;
        }
        
        .stChatMessage [data-testid="chatAvatarIcon-user"], 
        .stChatMessage [data-testid="chatAvatarIcon-assistant"] {
            display: none;
        }
        
        .stChatMessage [data-testid="chatMessage-user"] > div:first-child, 
        .stChatMessage [data-testid="chatMessage-assistant"] > div:first-child {
            background-color: transparent !important;
            border: none !important;
            padding: 0 !important;
        }
        
        /* Text styling */
        .user-message {
            font-size: 1.2em;
            font-weight: light;
            color: #F9F9F9;
            margin-bottom: 10px;
        }
        
        .pinyin-text {
            font-family: 'CustomChineseFont', sans-serif;
            font-size: 32px;
            transform: translateY(-28px);
        }
        
        .normal-text {
            font-family: sans-serif;
            font-size: 24px;
            line-height: 1.5;
            display: block;
            word-wrap: break-word;
            white-space: pre-wrap;
            max-width: 100%;
            align-items: center;
            transform: translateY(-17px);
        }
        
        .normal-text span {
            display: inline;
            vertical-align: baseline;
            margin-right: 1px; 
        }
        
        /* Button styling */
        .stButton > button {
            background-color: transparent !important;
            color: #f9f9f9 !important;
            border: none !important;
            text-align: left;
            text-decoration: none;
            cursor: pointer;
            font-size: 14px;
            padding: 0 !important;
            height: auto;
            line-height: 1;
            box-shadow: none !important;
            font-weight: normal !important;
            margin: 0 !important;
            min-width: 0 !important;
        }
        
        .stButton > button:hover {
            color: #ffffff !important;
            background-color: transparent !important;
            text-decoration: underline;
        }
        
        /* Layout styling */
        .row-widget.stHorizontal {
            flex-direction: row;
            justify-content: flex-start;
            gap: 10px;
        }
        
        .row-widget.stHorizontal > div {
            flex: 0 1 auto;
        }
        
        /* Markdown and text elements */
        .stMarkdown, .stMarkdown p {
            color: #f9f9f9 !important;
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: #f9f9f9 !important;
        }
        
        .element-container {
            background-color: transparent !important;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #f0f0f0;
        }
        
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] .stSelectbox,
        [data-testid="stSidebar"] .stSlider,
        [data-testid="stSidebar"] .stTextInput {
            color: black !important;
        }
        
        [data-testid="stSidebar"] .stMarkdown h1 {
            color: black !important;
        }
        
        [data-testid="stSidebar"] .stButton > button {
            color: black !important;
            background-color: #e0e0e0 !important;
            border: 1px solid #c0c0c0 !important;
            border-radius: 4px;
            padding: 5px 10px !important;
            text-align: center;
            font-weight: bold !important;
        }
        
        [data-testid="stSidebar"] .stButton > button:hover {
            background-color: #d0d0d0 !important;
            color: black !important;
            text-decoration: none;
        }
        </style>
    """, unsafe_allow_html=True)
    
    main()

if __name__ == "__main__":
    main()
