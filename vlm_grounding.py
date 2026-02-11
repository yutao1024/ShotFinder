import json
import os
import re
import random
import time
import llm_handler

def check_video_match(config, vlm_model, vlm_prompt, v_frames_dir, is_audio, num_frames, video_duration, content, logger):

    # === Read Audio Capability ===
    model_supports_audio = config.get('VLM_SUPPORTS_AUDIO', False)
    enable_audio_input = is_audio and model_supports_audio

    # === Prepare Prompt ===
    if enable_audio_input:
        prompt_text = vlm_prompt \
            .replace("$en_memory_data$", str(content)) \
            .replace("$VIDEO_DURATION$", str(video_duration)) \
            .replace("$NUM_FRAMES$", str(num_frames))
    else:
        prompt_text = vlm_prompt.replace("$en_memory_data$", str(content))
        # Replace placeholders for vision-only mode
        prompt_text = prompt_text.replace("$VIDEO_DURATION$", "unknown").replace("$NUM_FRAMES$", str(num_frames))

    # === Collect Frames ===
    def natural_key(s):
        return [int(t) if t.isdigit() else t for t in re.findall(r'\d+|\D+', s)]

    try:
        frame_files = sorted(
            [f for f in os.listdir(v_frames_dir) if f.lower().endswith(".jpg")],
            key=natural_key
        )
        if not frame_files:
            return {"frame_id": "N/A"}
            
        frame_paths = [os.path.join(v_frames_dir, f) for f in frame_files]
        
    except Exception as e:
        logger.error(f"[Grounding] Read frames failed: {e}")
        return {"frame_id": "N/A"}

    # Prepare Audio Path
    audio_path = os.path.join(v_frames_dir, "audio.wav") if enable_audio_input else None


    MAX_RETRIES = 5
    BASE_DELAY = 5
    MAX_DELAY = 60

    for attempt in range(MAX_RETRIES + 1):
        try:
            raw_content = ""

            # ================= DISPATCH TO HANDLER =================
            if "gemini" in vlm_model.lower():
                # Use Files API for Gemini
                raw_content = llm_handler.call_gemini_files_api(
                    config, vlm_model, prompt_text, frame_paths, audio_path, logger
                )
            
            elif "claude" in vlm_model.lower():
                # Use Beta Files API for Claude
                raw_content = llm_handler.call_claude_files_api(
                    config, vlm_model, prompt_text, frame_paths, audio_path, logger
                )
            
            else:
                # Use Base64 mode for GPT / Qwen
                raw_content = llm_handler.call_openai_base64(
                    config, vlm_model, prompt_text, frame_paths, audio_path, logger
                )
            # =======================================================

            # === Parse JSON Result ===
            try:
                # Clean Markdown tags
                cleaned_text = raw_content.replace("```json", "").replace("```", "").strip()
                result = json.loads(cleaned_text)
                
                frame_id = result.get("frame_id")
                if frame_id is not None and str(frame_id) != "N/A":
                    return {"frame_id": int(frame_id)}
                elif str(frame_id) == "N/A":
                    return {"frame_id": "N/A"}
                else:
                    logger.warning(f"[Grounding] Invalid output format: {result}. Retry...")
            
            except json.JSONDecodeError:
                logger.warning(f"[Grounding] JSON Parse Error. Raw: {raw_content[:100]}...")

            if attempt < MAX_RETRIES:
                 time.sleep(random.uniform(1.0, 3.0))
                 continue
            else:
                 return {"frame_id": "N/A"}

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate_limit" in error_str or "Quota" in error_str:
                if attempt < MAX_RETRIES:
                    sleep_time = min(MAX_DELAY, BASE_DELAY * (2 ** attempt)) + random.uniform(0.5, 2.0)
                    logger.warning(f"[Grounding] Rate Limit ({error_str[:50]}...). Retrying in {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
                    continue
            
            logger.error(f"[Grounding] Critical API Error: {e}")
            
            # If error is related to audio format support, try downgrading to vision-only
            if enable_audio_input and attempt < MAX_RETRIES:
                logger.warning("[Grounding] Retrying WITHOUT audio input.")
                audio_path = None # Disable audio for next retry
                enable_audio_input = False
                continue

            return {"frame_id": "N/A"}

    return {"frame_id": "N/A"}