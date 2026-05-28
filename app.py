import gradio as gr
import requests
import os
import json

# =========================================
# 🔑 OPENROUTER CONFIG
# =========================================

API_KEY = os.getenv("OPENROUTER_API_KEY")

API_URL = "https://openrouter.ai/api/v1/chat/completions"

MODEL = "qwen/qwen3-next-80b-a3b-instruct"


# =========================================
# 🧠 MEMORY LIMIT
# =========================================

def trim_history(history, max_turns=10):

    if history is None:
        return []

    return history[-max_turns:]


# =========================================
# 🔥 STREAMING CHAT FUNCTION
# =========================================

def chat(message, history):

    if history is None:
        history = []

    # empty message block
    if not message or not message.strip():
        yield "", history
        return

    # trim memory
    history = trim_history(history)

    # system prompt
    messages = [
        {
            "role": "system",
            "content": (
                "You are Sparks AI, an advanced AI assistant focused on "
                "robotics, electronics, IoT, automation, embedded systems, "
                "coding, AI, cybersecurity and modern technology. "
                "You also answer normal questions naturally. "
                "Give intelligent, structured, practical and clean responses."
            )
        }
    ]

    # history add
    for item in history:

        if (
            isinstance(item, dict)
            and "role" in item
            and "content" in item
        ):
            messages.append(item)

    # current user msg
    messages.append({
        "role": "user",
        "content": message
    })

    # add user msg instantly
    history.append({
        "role": "user",
        "content": message
    })

    # placeholder bot response
    history.append({
        "role": "assistant",
        "content": "⚡ Thinking..."
    })

    yield "", history

    # request headers
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://huggingface.co",
        "X-Title": "Sparks AI"
    }

    # request payload
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1200,
        "stream": True
    }

    try:

        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            stream=True,
            timeout=60
        )

        # HTTP errors
        if response.status_code != 200:

            try:
                error_data = response.json()

                error_msg = (
                    error_data.get("error", {})
                    .get("message", "Unknown API Error")
                )

            except:
                error_msg = "Unknown API Error"

            history[-1]["content"] = f"⚠️ {error_msg}"

            yield "", history
            return

        full_reply = ""

        # stream tokens
        for line in response.iter_lines():

            if not line:
                continue

            decoded = line.decode("utf-8")

            if not decoded.startswith("data: "):
                continue

            data_str = decoded[6:]

            # stream end
            if data_str == "[DONE]":
                break

            try:

                data_json = json.loads(data_str)

                choices = data_json.get("choices", [])

                if not choices:
                    continue

                delta = choices[0].get("delta", {})

                token = delta.get("content", "")

                if token:

                    full_reply += token

                    # live stream update
                    history[-1]["content"] = full_reply

                    yield "", history

            except:
                continue

        # empty fallback
        if not full_reply.strip():

            history[-1]["content"] = (
                "⚠️ Empty response from model."
            )

            yield "", history

    except Exception as e:

        history[-1]["content"] = (
            f"⚠️ Network Error: {str(e)}"
        )

        yield "", history


# =========================================
# 🔁 RETRY FUNCTION
# =========================================

def retry(history):

    if history is None:
        return history

    if len(history) < 2:
        return history

    last_user = None

    for i in range(len(history) - 1, -1, -1):

        item = history[i]

        if (
            isinstance(item, dict)
            and item.get("role") == "user"
        ):

            last_user = item.get("content")

            history = history[:i]

            break

    if not last_user:
        return history

    return chat(last_user, history)


# =========================================
# 🗑 CLEAR CHAT
# =========================================

def clear_chat():
    return []


# =========================================
# 🎨 UI
# =========================================

with gr.Blocks() as demo:

    # header
    gr.HTML("""

    <div style="
        text-align:center;
        padding-top:20px;
        padding-bottom:18px;
    ">

        <h1 style="
            font-size:42px;
            color:#ff9800;
            margin-bottom:10px;
            font-weight:700;
            letter-spacing:1px;
        ">
            ⚡ Sparks AI
        </h1>

        <p style="
            color:#8b949e;
            font-size:15px;
        ">
            Robotics • Electronics • IoT • Automation • Coding
        </p>

    </div>

    """)

    # chatbot
    chatbot = gr.Chatbot(height=620)

    # input row
    with gr.Row():

        msg = gr.Textbox(
            placeholder="Message Sparks AI...",
            show_label=False,
            scale=12,
            container=False
        )

        send = gr.Button(
            "➜",
            scale=1,
            min_width=70
        )

    # buttons
    with gr.Row():

        retry_btn = gr.Button("↻ Retry")

        clear_btn = gr.Button("✕ Clear")

    # =========================================
    # EVENTS
    # =========================================

    send.click(
        chat,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
    )

    msg.submit(
        chat,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
    )

    retry_btn.click(
        retry,
        inputs=chatbot,
        outputs=chatbot
    )

    clear_btn.click(
        clear_chat,
        outputs=chatbot
    )


# =========================================
# 🚀 LAUNCH
# =========================================

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,

    theme=gr.themes.Soft(
        primary_hue="orange",
        neutral_hue="slate"
    ),

    css="""

    body{
        background:#0b0f19;
    }

    .gradio-container{
        max-width:1200px !important;
        margin:auto !important;
        padding-top:10px !important;
    }

    /* USER MSG */

    .message.user{
        background:#1a1f2e !important;
        border:1px solid #2b3245 !important;
        border-radius:18px !important;
    }

    /* AI MSG */

    .message.bot{
        background:#121826 !important;
        border:1px solid #232c3d !important;
        border-radius:18px !important;
    }

    /* TEXTBOX */

    textarea{
        background:#111827 !important;
        color:white !important;
        border:1px solid #2d3748 !important;
        border-radius:18px !important;
        padding:16px !important;
        font-size:16px !important;
    }

    textarea:focus{
        border:1px solid #ff9800 !important;
        box-shadow:none !important;
    }

    /* BUTTONS */

    button{
        border-radius:16px !important;
        border:none !important;
        font-weight:600 !important;
        transition:0.2s ease !important;
    }

    button:hover{
        transform:scale(1.03);
    }

    /* CODE BLOCKS */

    pre{
        background:#0f172a !important;
        border-radius:14px !important;
        border:1px solid #263041 !important;
        padding:14px !important;
        overflow-x:auto !important;
    }

    code{
        color:#ffb74d !important;
    }

    /* SCROLLBAR */

    ::-webkit-scrollbar{
        width:8px;
    }

    ::-webkit-scrollbar-thumb{
        background:#2d3748;
        border-radius:10px;
    }

    /* FOOTER */

    footer{
        display:none !important;
    }

    """
)
