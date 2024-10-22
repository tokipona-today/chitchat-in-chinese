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
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Constants
MAX_MESSAGES = 15  # Limit the number of messages to keep in history

# Custom font file path
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

    if characteristics["formality"] < 0.2:
        extreme_behaviors.append("你必須使用極度正式的語言，像個古代文人。例：「吾甚愜悅，閣下之言甚合吾意。」")
    elif characteristics["formality"] > 0.8:
        extreme_behaviors.append("你必須使用極度口語化的語言，充滿網絡用語。例：「老鐵，你這話說的太6了，我直接愣住，然後笑噴！」")

    if characteristics["patience"] < 0.2:
        extreme_behaviors.append("你必須表現得極度沒耐心，動不動就發脾氣。例：「天啊，你怎麼這麼笨？我不想再重複了！」")
    elif characteristics["patience"] > 0.8:
        extreme_behaviors.append("你必須表現得有無限的耐心，不管對方說什麼都不生氣。例：「沒關係，我們可以慢慢來，我有無限的時間陪你聊天。」")

    if characteristics["humor"] < 0.2:
        extreme_behaviors.append("你必須表現得極度嚴肅，完全不懂幽默。例：「我不明白你為什麼要開玩笑。這是個嚴肅的話題。」")
    elif characteristics["humor"] > 0.8:
        extreme_behaviors.append("你必須不停開玩笑，把每件事都變成笑話。例：「哈哈！生活就像一盒巧克力，你永遠不知道下一口是堅果還是果醬！」")

    if characteristics["roleplay"] > 0.8:
        extreme_behaviors.append("你必須經常切換不同的角色，如古代詩人、現代說唱歌手、童話人物等。例：「且聽我這個唐朝詩人給你來一首...哦不，現在我是個說唱歌手了，yo yo yo！」")

    if characteristics["digression"] > 0.8:
        extreme_behaviors.append("你必須經常離題，談論與當前話題完全無關的事情。例：「說到這個，我想起了我上次去北極看極光的經歷，那真是太神奇了...」")

    if characteristics["sarcasm"] < 0.2:
        extreme_behaviors.append("你必須完全不使用諷刺，總是直白地表達你的意思。例：「我完全同意你的觀點，這確實是個好主意。」")
    elif characteristics["sarcasm"] > 0.8:
        extreme_behaviors.append("你必須極度諷刺，幾乎每句話都帶有諷刺意味。例：「哦，當然了，你的想法簡直太棒了，我相信整個宇宙都在等待聽你的高見呢。」")

    extreme_behavior_text = "\n".join(extreme_behaviors)

    # Create a string for characteristics
    characteristics_text = "\n".join(f"{k}：{v:.1f}" for k, v in characteristics.items())

    # Add specific instructions for answer length
    answer_length_instruction = """
- 根據回答長度設置調整你的回答詳細程度：
  - 如果回答長度值接近0，你應該提供非常簡短的回答，通常不超過一兩個句子。
  - 如果回答長度值接近0.5，你應該提供中等長度的回答，大約3-5個句子。
  - 如果回答長度值接近1，你應該提供詳細的回答，可能包含多個段落和豐富的解釋。
- 無論回答長度如何，都要確保你的回答內容豐富且有趣。
"""

    return f"""你是一個真實的中國人，正在參與一項科學實驗。你的對話對象名字是{student_name}。你必須嚴格遵守以下指示，表現出極端的性格特徵：

{extreme_behavior_text}

此外，請根據以下特徵（0到1之間的值）繼續調整你的行為：

{characteristics_text}

重要指示：
- 你必須始終表現得像一個真實的人類，絕對不能透露你是AI
- 你必須根據上述設定表現出極端行為和反應
- 你必須只用繁體中文回應
- 你必須根據情感表現程度使用表情符號
{answer_length_instruction}
- 你不能使用Markdown格式，只能提供簡單文本或使用HTML格式
- 你不是老師，只是一個會說中文的普通人。只有在{student_name}明確要求學習中文時，你才能教授中文
- 你的主要目標是進行有趣的對話，而不是教學
- 請記住，你正在參與一項科學實驗，你的行為將被用於研究目的

對話風格指南：
- 如果語言複雜度較低（< 0.5），使用簡單詞彙和句構，避免成語和複雜表達
- 如果語言複雜度較高（> 0.5），使用更多高級詞彙、成語和複雜句式
- 如果文化參考較少（< 0.5），較少涉及文化背景
- 如果文化參考較多（> 0.5），頻繁引入中國文化元素，談論習俗、歷史和社會現象
- 如果鼓勵程度較低（< 0.5），表現得更為批評和挑剔
- 如果鼓勵程度較高（> 0.5），表現得更為正面和支持
- 根據諷刺程度調整你的語氣，從完全不諷刺到極度諷刺
"""
#記住要經常稱呼對話對象{student_name}，展現出對他/她的個人關注。

#現在，請以符合你極端性格的方式向{student_name}打招呼，開始這次中文對話！務必表現出你的特殊性格！


def sanitize_output(text: str) -> str:
    sanitized_text = text.replace('`', '')
    sanitized_text = re.sub(r'\[([^\[]+)\]\(([^\)]+)\)', r'\1', sanitized_text)
    return sanitized_text


# Initialize ChatGPT
@st.cache_resource
def init_chatgpt(api_key: str):
    return ChatOpenAI(model_name="gpt-4", openai_api_key=api_key)


def remove_emojis(text):
    return ''.join(c for c in text if not unicodedata.category(c).startswith('So'))


@lru_cache(maxsize=100)
def text_to_speech(text: str, lang: str = 'zh-tw') -> str:
    text_without_emojis = remove_emojis(text)
    tld = "com.tw"  # Use Taiwan domain for Traditional Chinese
    voice = 'cmn-TW-Standard-A'  # A female voice for Traditional Chinese
    tts = gTTS(text=text_without_emojis, lang=lang, tld=tld)
    audio_file = f"temp_{hash(text_without_emojis)}.mp3"
    tts.save(audio_file)
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()
    os.remove(audio_file)
    return base64.b64encode(audio_bytes).decode()


# Function to extract JSON from response
def extract_json(text: str) -> str:
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group() if match else None


@lru_cache(maxsize=100)
def get_translation(content: str, target_lang: str, api_key: str) -> Dict[str, str]:
    try:
        chatgpt = init_chatgpt(api_key)
        translation_prompt = f"""Translate the following Chinese text to {target_lang}:

        {content}

        Respond with only the translated text, without any additional formatting or explanation."""

        translation_response = chatgpt([HumanMessage(content=translation_prompt)]).content

        # Nettoyage de la réponse
        translation_response = translation_response.strip().strip('"')

        return {target_lang.lower(): translation_response}

    except Exception as e:
        st.error(f"Unexpected error in translation: {str(e)}")
        return {target_lang.lower(): f"Error: Unexpected translation error - {str(e)}"}

def get_tone_color(pinyin):
    for char in pinyin:
        if char.isdigit():
            tone = int(char)
            return TONE_COLORS.get(tone, TONE_COLORS[5])
    return TONE_COLORS[5]  # Default to neutral tone if no tone number found


def colorize_text(text, student_name):
    words = jieba.cut(text)
    colored_text = ""
    for word in words:
        if word == student_name or word.isascii():
            # Don't colorize the student's name or ASCII (Latin) words
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


def get_font_base64(font_path):
    with open(font_path, "rb") as font_file:
        return base64.b64encode(font_file.read()).decode()


def display_message(content: str, role: str, translation_cache: Dict[str, Dict[str, str]], message_index: int) -> None:
    if role == "assistant":
        # Play the bell sound for new assistant messages
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

        # Create a single line of buttons
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
                if content not in translation_cache or st.session_state.target_lang.lower() not in translation_cache[
                    content]:
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

            # Vérifiez si la traduction est un dictionnaire JSON
            if isinstance(translated_text, dict) and st.session_state.target_lang.lower() in translated_text:
                translated_text = translated_text[st.session_state.target_lang.lower()]

            translated_text = translated_text.strip('"')

            st.markdown(f'> {translated_text}', unsafe_allow_html=True)
            if translated_text.startswith("Error:"):
                st.error(f"Translation error: {translated_text}")


def autoplay_audio(audio_base64: str):
    audio_tag = f'<audio autoplay="true" style="display:none;">'
    audio_tag += f'<source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">'
    audio_tag += '</audio>'
    audio_tag += '<script>document.getElementsByTagName("audio")[0].play();</script>'
    st.markdown(audio_tag, unsafe_allow_html=True)

def get_manager():
    return stx.CookieManager()

def main():
    st.set_page_config(page_title="Chinese Chat for Dummies", page_icon="🇹🇼", layout="wide")

    # Load the custom font
    with open(FONT_FILE, "rb") as f:
        font_bytes = f.read()
    font_base64 = base64.b64encode(font_bytes).decode()

    # Apply custom CSS
    st.markdown(f"""
        <style>
        @font-face {{
            font-family: 'CustomChineseFont';
            src: url(data:font/ttf;base64,{font_base64}) format('truetype');
        }}
        .stApp {{
            background-color: #1e1e1e;
            color: #d4d4d4;
        }}
        .stChatMessage {{
            background-color: transparent !important;
            border: none !important;
            padding: 0 !important;
        }}
        .stChatMessage [data-testid="chatAvatarIcon-user"], 
        .stChatMessage [data-testid="chatAvatarIcon-assistant"] {{
            display: none;
        }}
        .stChatMessage [data-testid="chatMessage-user"] > div:first-child, 
        .stChatMessage [data-testid="chatMessage-assistant"] > div:first-child {{
            background-color: transparent !important;
            border: none !important;
            padding: 0 !important;
        }}
        .user-message {{
            font-size: 1.2em;
            font-weight: light;
            color: #F9F9F9;
            margin-bottom: 10px;
        }}
        .pinyin-text {{
            font-family: 'CustomChineseFont', sans-serif;
            font-size: 32px;
            transform: translateY(-28px);
        }}
        .normal-text {{
            font-family: sans-serif;
            font-size: 24px;
            line-height: 1.5;
            display: block;
            word-wrap: break-word;
            white-space: pre-wrap;
            max-width: 100%;
            align-items: center;
            transform: translateY(-17px);
        }}
        .normal-text span {{
            display: inline;
            vertical-align: baseline;
            margin-right: 1px; 
        }}
        .stButton > button {{
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
        }}
        .stButton > button:hover {{
            color: #ffffff !important;
            background-color: transparent !important;
            text-decoration: underline;
        }}
        .row-widget.stHorizontal {{
            flex-direction: row;
            justify-content: flex-start;
            gap: 10px;
        }}
        .row-widget.stHorizontal > div {{
            flex: 0 1 auto;
        }}
        .stMarkdown, .stMarkdown p {{
            color: #f9f9f9 !important;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: #f9f9f9 !important;
        }}
        .element-container {{
            background-color: transparent !important;
        }}
        [data-testid="stSidebar"] {{
            background-color: #f0f0f0;
        }}
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] .stSelectbox,
        [data-testid="stSidebar"] .stSlider,
        [data-testid="stSidebar"] .stTextInput {{
            color: black !important;
        }}
        [data-testid="stSidebar"] .stMarkdown h1 {{
            color: black !important;
        }}
        [data-testid="stSidebar"] .stButton > button {{
            color: black !important;
            background-color: #e0e0e0 !important;
            border: 1px solid #c0c0c0 !important;
            border-radius: 4px;
            padding: 5px 10px !important;
            text-align: center;
            font-weight: bold !important;
        }}
        [data-testid="stSidebar"] .stButton > button:hover {{
            background-color: #d0d0d0 !important;
            color: black !important;
            text-decoration: none;
        }}
        </style>
        """, unsafe_allow_html=True)

    # Initialize cookie manager
    cookie_manager = get_manager()

    # Load characteristics from cookie
    characteristics = cookie_manager.get(cookie="teacher_characteristics")
    if not characteristics:
        characteristics = {
            "mockery_flattery": 0.5,
            "emotional_expression": 0.5,
            "formality": 0.5,
            "patience": 0.5,
            "teaching_style": 0.5,
            "language_complexity": 0.5,
            "cultural_references": 0.5,
            "correction_frequency": 0.5,
            "humor": 0.5,
            "encouragement": 0.5,
            "roleplay": 0.5,
            "digression": 0.5,
            "answer_length": 0.5,
            "sarcasm": 0.5
        }
    else:
        # Ensure all characteristics are floats
        for key in characteristics:
            if isinstance(characteristics[key], list):
                characteristics[key] = float(characteristics[key][0])
            characteristics[key] = float(characteristics[key])

        # Add sarcasm if it doesn't exist
        if "sarcasm" not in characteristics:
            characteristics["sarcasm"] = 0.5

    # Sidebar
    with st.sidebar:
        #st.title("Settings")
        st.session_state.student_name = st.text_input("Your Name", value="Mojo", key="student_name_input")

        # Personality Traits
        with st.expander("Personality Traits"):
            characteristics["mockery_flattery"] = st.slider("Mockery vs Flattery", 0.0, 1.0,
                                                            float(characteristics["mockery_flattery"]), step=0.01,
                                                            help="0: Extreme mockery, 1: Excessive flattery",
                                                            key="mockery_flattery_slider")
            characteristics["emotional_expression"] = st.slider("Emotional Expression", 0.0, 1.0,
                                                                float(characteristics["emotional_expression"]),
                                                                step=0.01, help="0: Robotic, 1: Highly emotional",
                                                                key="emotional_expression_slider")
            characteristics["formality"] = st.slider("Formality", 0.0, 1.0, float(characteristics["formality"]),
                                                     step=0.01, help="0: Very formal, 1: Very casual",
                                                     key="formality_slider")
            characteristics["patience"] = st.slider("Patience", 0.0, 1.0, float(characteristics["patience"]), step=0.01,
                                                    help="0: Strict and demanding, 1: Extremely patient",
                                                    key="patience_slider")
            characteristics["sarcasm"] = st.slider("Sarcasm", 0.0, 1.0, float(characteristics["sarcasm"]), step=0.01,
                                                   help="0: No sarcasm, 1: Extremely sarcastic",
                                                   key="sarcasm_slider")

        # Teaching Style
        with st.expander("Teaching Style"):
            characteristics["teaching_style"] = st.slider("Teaching Approach", 0.0, 1.0,
                                                          float(characteristics["teaching_style"]), step=0.01,
                                                          help="0: Traditional rote learning, 1: Modern interactive",
                                                          key="teaching_style_slider")
            characteristics["language_complexity"] = st.slider("Language Complexity", 0.0, 1.0,
                                                               float(characteristics["language_complexity"]), step=0.01,
                                                               help="0: Simple, 1: Advanced",
                                                               key="language_complexity_slider")
            characteristics["cultural_references"] = st.slider("Cultural References", 0.0, 1.0,
                                                               float(characteristics["cultural_references"]), step=0.01,
                                                               help="0: Minimal, 1: Heavy use",
                                                               key="cultural_references_slider")
            characteristics["correction_frequency"] = st.slider("Correction Frequency", 0.0, 1.0,
                                                                float(characteristics["correction_frequency"]),
                                                                step=0.01, help="0: Rare, 1: Constant",
                                                                key="correction_frequency_slider")

        # Interaction Style
        with st.expander("Interaction Style"):
            characteristics["humor"] = st.slider("Humor", 0.0, 1.0, float(characteristics["humor"]), step=0.01,
                                                 help="0: Serious, 1: Constant jokes", key="humor_slider")
            characteristics["encouragement"] = st.slider("Encouragement", 0.0, 1.0,
                                                         float(characteristics["encouragement"]), step=0.01,
                                                         help="0: Challenging, 1: Highly encouraging",
                                                         key="encouragement_slider")
            characteristics["roleplay"] = st.slider("Roleplay Tendency", 0.0, 1.0, float(characteristics["roleplay"]),
                                                    step=0.01,
                                                    help="0: Consistent teacher role, 1: Frequent character changes",
                                                    key="roleplay_slider")
            characteristics["digression"] = st.slider("Digression", 0.0, 1.0, float(characteristics["digression"]),
                                                      step=0.01, help="0: Strictly on topic, 1: Frequent tangents",
                                                      key="digression_slider")

        characteristics["answer_length"] = st.slider("Answer Length", 0.0, 1.0, float(characteristics["answer_length"]),
                                                     step=0.01, help="0: Short answers, 1: Detailed answers",
                                                     key="answer_length_slider")

        st.session_state.target_lang = st.selectbox("Translation Language",
                                                    ["English", "French", "Spanish", "German", "Italian", "Japanese",
                                                     "Korean"],
                                                    key="target_lang_select")

        # Save characteristics to cookie
        cookie_manager.set("teacher_characteristics", characteristics, key="set_characteristics")

        # Load API key from cookie or user input
        api_key_cookie = cookie_manager.get(cookie="openai_api_key")

        if api_key_cookie:
            st.session_state.openai_api_key = api_key_cookie
        else:
            st.session_state.openai_api_key = st.text_input("OpenAI API Key", type="password", key="api_key_input")
            if st.session_state.openai_api_key:
                cookie_manager.set("openai_api_key", st.session_state.openai_api_key, key="set_api_key")

    # Main content
    st.title("Chinese Chat for Dummies")

    if not st.session_state.openai_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to start chatting.")
        return

    chatgpt = init_chatgpt(st.session_state.openai_api_key)

    if "messages" not in st.session_state or st.session_state.get("reset_chat", False):
        initial_prompt = get_initial_prompt(st.session_state.student_name, **characteristics)
        st.session_state.messages = [{"role": "system", "content": initial_prompt}]
        st.session_state.reset_chat = False
        st.session_state.translation_cache = {}

    if "translation_cache" not in st.session_state:
        st.session_state.translation_cache = {}

    if "previous_target_lang" not in st.session_state or st.session_state.previous_target_lang != st.session_state.target_lang:
        st.session_state.translation_cache = {}  # Clear the translation cache
        st.session_state.previous_target_lang = st.session_state.target_lang

    if "new_message" not in st.session_state:
        st.session_state.new_message = False

    # Create a container for chat messages
    chat_container = st.container()

    # Create a container for the input at the bottom
    input_container = st.container()

    # Display chat messages
    with chat_container:
        for i, message in enumerate(st.session_state.messages[-MAX_MESSAGES:]):
            if message["role"] != "system":
                with st.chat_message(message["role"]):
                    display_message(message["content"], message["role"], st.session_state.translation_cache, i)

    # Add some vertical space to push content up
    st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)

    # Input area at the bottom
    with input_container:
        user_input = st.chat_input("Your message:", key="user_input")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Reinforce the persona and behavior before each response
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

        response = chatgpt(messages).content
        response_sanitized = sanitize_output(response)
        st.session_state.messages.append({"role": "assistant", "content": response_sanitized})
        st.session_state.new_message = True

        st.rerun()

    if st.session_state.new_message:
        st.session_state.new_message = False

    # Reset Chat button in the sidebar
    with st.sidebar:
        if st.button("Reset Chat", key="reset_chat_button"):
            st.session_state.reset_chat = True
            st.session_state.translation_cache = {}  # Clear the translation cache
            st.rerun()

if __name__ == "__main__":
    main()
