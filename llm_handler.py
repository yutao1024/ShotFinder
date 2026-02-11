import os
import json
import time
import base64
import logging
from typing import List, Dict, Any

# SDK Imports
import google.generativeai as genai
from google.generativeai import types
from openai import OpenAI
from anthropic import Anthropic

# ================== Helper Functions ==================

def encode_image_base64(image_path):
    """Encodes an image file to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def encode_audio_base64(audio_path):
    """Encodes an audio file to a base64 string."""
    with open(audio_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode('utf-8')

# ================== Model Specific Implementations ==================

def call_gemini_files_api(config, model_name, prompt_text, frame_paths, audio_path, logger):
    """
    Gemini Specific: Use Files API to upload large amounts of images/audio -> Get references -> Inference -> Delete files.
    """
    genai.configure(api_key=config['GEMINI_API_KEY'])
    uploaded_files = []
    
    try:
        # 1. Batch upload images
        for p in frame_paths:
            # Using SDK to upload
            f = genai.upload_file(path=p, mime_type="image/jpeg")
            uploaded_files.append(f)
        
        # 2. Upload audio (if exists)
        if audio_path and os.path.exists(audio_path):
            f_audio = genai.upload_file(path=audio_path, mime_type="audio/wav")
            uploaded_files.append(f_audio)

        # 3. Wait for files to be processed (Gemini Files API is asynchronous)
        # Simple polling check; images are usually fast, audio/video might take longer
        for f in uploaded_files:
            while f.state.name == "PROCESSING":
                time.sleep(1)
                f = genai.get_file(f.name)
            if f.state.name == "FAILED":
                raise ValueError(f"File upload failed: {f.name}")

        # 4. Construct request content
        # In Files API mode, the contents list directly contains file objects + prompt
        contents = list(uploaded_files) + [prompt_text]

        # 5. Call the model
        logger.info(f"[Gemini] Calling model {model_name} with {len(uploaded_files)} files...")
        client = genai.GenerativeModel(model_name)
        response = client.generate_content(
            contents,
            generation_config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json"
            )
        )
        return response.text

    finally:
        # 6. Clean up remote files (Critical to avoid quota usage)
        logger.info("[Gemini] Cleaning up remote files...")
        for f in uploaded_files:
            try:
                genai.delete_file(f.name)
            except Exception as e:
                logger.warning(f"[Gemini] Failed to delete file {f.name}: {e}")


def call_claude_files_api(config, model_name, prompt_text, frame_paths, audio_path, logger):
    """
    Claude Specific: Use Beta Files API.
    Upload -> Get file_id -> Reference file_id in message -> DELETE files.
    """
    # Initialize client with Beta Header
    client = Anthropic(
        api_key=config['ANTHROPIC_API_KEY'],
        default_headers={"anthropic-beta": "files-api-2025-04-14"}
    )

    message_content = []
    uploaded_file_ids = []  # [新增] 用于记录已上传的 File ID

    try:
        # 1. Upload images and get IDs
        for p in frame_paths:
            with open(p, "rb") as f:
                try:
                    file_content = f.read()
                    # Call Files API to upload
                    uploaded = client.beta.files.create(
                        file=(os.path.basename(p), file_content, "image/jpeg"),
                        purpose="message_attachment"
                    )
                    file_id = uploaded.id
                    
                    # [新增] 记录 ID 以便后续删除
                    uploaded_file_ids.append(file_id)
                    
                    # Add to message body
                    message_content.append({
                        "type": "image",
                        "source": {
                            "type": "file",
                            "file_id": file_id
                        }
                    })
                except Exception as e:
                    logger.error(f"[Claude] File upload failed for {p}: {e}")
                    raise e

        # Claude API logic for audio omitted as per original code...

        # Add text prompt
        message_content.append({"type": "text", "text": prompt_text})

        # 2. Send message
        logger.info(f"[Claude] Sending message with {len(frame_paths)} file references...")
        response = client.messages.create(
            model=model_name,
            messages=[{"role": "user", "content": message_content}],
            max_tokens=4096,
            temperature=0.0
        )
        return response.content[0].text

    finally:
        # 3. Clean up remote files
        if uploaded_file_ids:
            logger.info(f"[Claude] Cleaning up {len(uploaded_file_ids)} remote files...")
            for fid in uploaded_file_ids:
                try:
                    client.beta.files.delete(file_id=fid)
                except Exception as e:
                    # 删除失败不应中断程序，记录警告即可
                    logger.warning(f"[Claude] Failed to delete file {fid}: {e}")


def call_openai_base64(config, model_name, prompt_text, frame_paths, audio_path, logger):
    """
    GPT / Qwen (OpenAI Compatible Interface) Specific: Full Base64 Injection.
    """
    # Distinguish between Qwen (DashScope) and GPT (OpenAI)
    if "qwen" in model_name.lower():
        client = OpenAI(
            api_key=config['DASHSCOPE_API_KEY'],
            base_url=config.get('DASHSCOPE_BASE_URL')
        )
    else:
        client = OpenAI(api_key=config['OPENAI_API_KEY'])

    messages_content = [{"type": "text", "text": prompt_text}]

    # 1. Convert Images to Base64
    for p in frame_paths:
        b64_img = encode_image_base64(p)
        messages_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
        })

    # 2. Convert Audio to Base64 (If supported)
    if audio_path and os.path.exists(audio_path):
        b64_audio = encode_audio_base64(audio_path)
        messages_content.append({
            "type": "input_audio", 
            "input_audio": {
                "data": b64_audio, 
                "format": "wav"
            }
        })

    # 3. Send Request
    kwargs = {
        "model": model_name,
        "messages": [{"role": "user", "content": messages_content}],
        "temperature": 0.0
    }

    # JSON Mode compatibility handling for Qwen
    # Only enable response_format if it's NOT Qwen (or if it's a model known to support it like Omni)
    # Adjust logic based on specific Qwen version capabilities if needed.
    if "qwen" not in model_name.lower() or "omni" in model_name.lower():
        kwargs["response_format"] = {"type": "json_object"}

    logger.info(f"[OpenAI/Qwen] Sending request with {len(frame_paths)} images (Base64)...")
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content