import re
import json
import random
import time
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from openai import OpenAI
from anthropic import Anthropic
from tools_lib import search_videos


def parse_response(content):
    think = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    tool = re.search(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)

    tool_data = None
    if tool:
        try:
            clean_json = tool.group(1).strip().replace("```json", "").replace("```", "")
            tool_data = json.loads(clean_json)
        except: pass

    return {
        "think": think.group(1).strip() if think else None,
        "tool_call": tool_data
    }


def get_safe_response_text(response):
    try:
        if not response.candidates:
            return None, "No candidates returned"
        
        candidate = response.candidates[0]

        if not candidate.content or not candidate.content.parts:
            if candidate.finish_reason == 3: # SAFETY
                return None, f"Blocked by Safety Filters: {candidate.safety_ratings}"

            return None, f"Empty Response (Finish Reason: {candidate.finish_reason})"
        
        return response.text, None
        
    except ValueError as e:
        return None, str(e)
    except Exception as e:
        return None, str(e)


def run_agent_search(task_id, user_content_str, config, prompt, is_resolution, logger):
    
    model_name = config['MODEL_NAME']
    
    # === Identify Client Type ===
    client_type = "gemini"
    if "gpt" in model_name.lower() or "qwen" in model_name.lower():
        client_type = "openai"
    elif "claude" in model_name.lower():
        client_type = "anthropic"

    # === Initialize Client ===
    client = None
    chat_gemini = None

    if client_type == "gemini":
        genai.configure(api_key=config['GEMINI_API_KEY'])
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=prompt, 
            safety_settings=safety_settings
        )
        chat_gemini = model.start_chat(history=[])
    
    elif client_type == "openai":
        # Use DashScope config if Qwen is detected
        if "qwen" in model_name.lower():
            client = OpenAI(
                api_key=config['DASHSCOPE_API_KEY'], 
                base_url=config.get('DASHSCOPE_BASE_URL')
            )
        else:
            client = OpenAI(api_key=config['OPENAI_API_KEY'])

    elif client_type == "anthropic":
        client = Anthropic(api_key=config['ANTHROPIC_API_KEY'])

    current_input_text = user_content_str
    found_urls = []

    logger.info(f"=====> Agent Start Search: {task_id} using {model_name} ({client_type})")
    logger.info(f"====================== Starting search ============================")
    logger.info(f"===> [LLM INPUT Context] ===>")
    logger.info(current_input_text)
    logger.info("=" * 80)

    MAX_RETRIES = 5          
    BASE_DELAY = 5           
    MAX_DELAY = 60          

    time.sleep(random.uniform(0.1, 1.0))

    for attempt in range(MAX_RETRIES + 1):

        try:
            content = None
            error_msg = None

            # === Execute LLM Call ===
            if client_type == "gemini":
                response = chat_gemini.send_message(current_input_text)
                content, error_msg = get_safe_response_text(response)
            
            elif client_type == "openai":
                try:
                    resp = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": current_input_text}
                        ],
                        temperature=0.1
                    )
                    content = resp.choices[0].message.content
                except Exception as e:
                    error_msg = str(e)

            elif client_type == "anthropic":
                try:
                    resp = client.messages.create(
                        model=model_name,
                        system=prompt,
                        messages=[{"role": "user", "content": current_input_text}],
                        max_tokens=4096,
                        temperature=0.1
                    )
                    content = resp.content[0].text
                except Exception as e:
                    error_msg = str(e)

            if not content:
                logger.warning(f"[Search] Response Error: {error_msg}")
                if error_msg and ("429" in error_msg or "rate" in error_msg.lower()):
                    raise Exception(error_msg)
                
                if attempt < MAX_RETRIES:
                    continue
                else:
                    return []

            logger.info(f"===> [LLM OUTPUT Context] ===>")
            logger.info(content)
            logger.info("=" * 80)

            parsed = parse_response(content)
            
            if parsed['tool_call']:
                name = parsed['tool_call']['name']
                args = parsed['tool_call']['arguments']
                
                if name == "search_videos":
                    found_urls = search_videos(config=config, queries = args.get('query'), serpapi_key = config['SERPAPI_KEY'], is_resolution=is_resolution, logger = logger)

            break

        except Exception as e:
            error_str = str(e)
            
            if "429" in error_str or "ResourceExhausted" in error_str or "Quota" in error_str or "rate_limit" in error_str:
                if attempt < MAX_RETRIES:
                    sleep_time = min(MAX_DELAY, BASE_DELAY * (2 ** attempt))
                    sleep_time += random.uniform(0.5, 2.0)
                    
                    logger.warning(f"[Search] Rate Limit Hit (429). Retrying in {sleep_time:.2f}s... (Attempt {attempt+1}/{MAX_RETRIES}) | Task: {task_id}")
                    time.sleep(sleep_time)
                    continue
                else:
                    logger.error(f"[Search] Max retries exceeded for Task {task_id}.")
                    return []

            else:
                logger.error(f"[Search] Agent Error: {e}")
                return []

    return found_urls