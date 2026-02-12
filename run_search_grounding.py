import os
import json
import time
import random
import hashlib
import subprocess
import shutil
from google import genai
from google.genai import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from agent_search import run_agent_search
from tools_lib import format_user_input, parse_timestamp, download_videos_deterministic
from vlm_grounding import check_video_match

def get_video_info(file_path):

    cmd = [
        'ffprobe', 
        '-v', 'error', 
        '-select_streams', 'v:0', 
        '-show_entries', 'stream=duration', 
        '-of', 'json', 
        file_path
    ]
    try:

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
        info = json.loads(result.stdout)
        stream = info['streams'][0]
        return float(stream.get('duration', 0))
    except Exception as e:
        print(f"Error reading video info: {e}")
        return None


def run_search(config, data, search_prompt, logger):
    task_id = data['id']

    en_content, audio, resolution = format_user_input(data)

    formatted_input = f"Please find the link to this video: {en_content}"

    target_urls = run_agent_search(task_id = task_id, user_content_str = formatted_input, config = config, prompt = search_prompt, is_resolution = resolution,logger = logger)

    if target_urls:
        target_urls.append(data['Video URL'])
        return target_urls, audio, resolution, en_content

    return [], False, False


def download_videos(config, data, target_urls, is_resolution, logger):

    task_id = data['id']

    save_dir = os.path.join(config['OUTPUT_DIR'], task_id)
    os.makedirs(save_dir, exist_ok=True)
    meta_path = os.path.join(save_dir, "metadata.json")

    if target_urls:
        results, is_gt_video = download_videos_deterministic(config = config, url_list = target_urls, save_dir = save_dir, 
                                                cookies_path = config['COOKIES_FILE'], is_resolution = is_resolution,logger=logger)

        if is_gt_video:
            with open(meta_path, 'w') as fout:
                json.dump(results, fout, indent=4, ensure_ascii=False)
            logger.info(f"[Downloading] Task Finished. Saved {len(results)} videos and 1 GT video.")
        else:
            with open(meta_path, 'w') as fout: 
                json.dump([], fout)
            logger.info(f"[Downloading] Error with GT video！Task Finished. Saved {len(results)} videos. ")
    else:
        logger.info("[Downloading] No videos downloaded.")

        with open(meta_path, 'w') as fout: 
            json.dump([], fout)


def extract_uniform_frames(video_path, output_dir, num_frames, video_id, src_duration, logger):
    try:
        os.makedirs(output_dir, exist_ok=True)

        interval = src_duration / num_frames
    
        for i in range(num_frames):
  
            t = (i + 0.5) * interval

            seek_t = max(t - 0.5, 0.0)

            out_path = os.path.join(output_dir, f"{i:03d}.jpg")

            cmd = [
                "ffmpeg",
                "-y",
                "-loglevel", "error",
                "-err_detect", "ignore_err",
                "-ss", f"{seek_t:.3f}",  
                "-i", video_path,
                "-frames:v", "1",        
                "-q:v", "7",             
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuvj420p", 
                out_path
            ]

            subprocess.run(cmd, check=True)


        logger.info(f"[Extracting frames] Finish video: {video_id:<60} | Extract: {num_frames} frames.")

    except subprocess.CalledProcessError as e:
        logger.error(f"[Extracting frames] Failed extracting {video_id}: {e}")
    except Exception as e:
        logger.error(f"[Extracting frames] Error: {e}")


def extract_audio(video_path, audio_path, logger):

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-i", video_path,
        "-vn",              
        audio_path
    ]

    try:
        subprocess.run(cmd, check=True)
        logger.info(f"[Audio] Saved audio to {audio_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"[Audio] Extract failed: {video_path}, {e}")
        return False


def process_videos(frames_dir, vid, config, is_audio, logger):

    meta_path = os.path.join(config['OUTPUT_DIR'], vid, "metadata.json")

    with open(meta_path, 'r') as f:
        downloaded = json.load(f)


    if not downloaded:
        logger.warning(f"[Process] No videos to process for {vid}. Skipping.")
  
        index_path = os.path.join(frames_dir, "frames_index.json")
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return 

    def process_one_video(v):
        if not os.path.exists(v['path']):
            logger.warning(f"[Process] Missing file: {v['path']}")
            return None

        src_duration = get_video_info(v['path'])
        if not src_duration:
            logger.error(f"[Process] Cannot read source: {v['path']}")
            return None

        title_clean = "".join([c for c in v['title'] if c.isalnum()])[:10]
        title_hash = hashlib.md5(v['title'].encode("utf-8")).hexdigest()[:4]
        safe_title = f"{title_clean}_{title_hash}"
        v_frames_dir = os.path.join(frames_dir, safe_title)
        os.makedirs(v_frames_dir, exist_ok=True)

        if src_duration < 180:
            frame_num = 32
        elif src_duration < 600:
            frame_num = 64
        elif src_duration < 1800:
            frame_num = 128
        else:
            frame_num = 192

        extract_uniform_frames(
            video_path=v['path'],
            output_dir=v_frames_dir,
            num_frames=frame_num,
            video_id=v["title"],
            src_duration=src_duration,
            logger=logger
        )

        audio_path = None
        if is_audio:
            audio_path = os.path.join(v_frames_dir, "audio.wav")
            extract_audio(v['path'], audio_path, logger)

        return {
            'title': v['title'],
            'dir_path': v_frames_dir,
            'audio_path': audio_path,
            "frame_num": frame_num,
            "src_duration":src_duration
        }

    max_workers = min(2, len(downloaded))
    frames_info = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_one_video, v) for v in downloaded]

        for future in as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    frames_info.append(result)
            except Exception as e:
                logger.error(f"[Process] Worker failed: {e}")

    index_path = os.path.join(frames_dir, "frames_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(frames_info, f, ensure_ascii=False, indent=2)

    logger.info(f"[Process] Saved frame index to {index_path}")



def run_grounding(config, frames_dir, res_dir, data, prompt, is_audio, content, logger):

    vid_id = data['id']
            
    res_path = os.path.join(res_dir, f"{vid_id}_result.json")

    index_path = os.path.join(frames_dir, "frames_index.json")

    if is_audio:
        match_prompt = prompt['MATCH_PROMPT']
    else:
        match_prompt = prompt['AUDIO_MATCH_PROMPT']

    with open(index_path, 'r') as f: 
        candidates = json.load(f)
    
    if not candidates:
        logger.warning(f"[Grounding] No candidate frames for {vid_id}. Skipping grounding.")
        with open(res_path, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=4, ensure_ascii=False)
        return

    def evaluate_one_video(idx, vid):
        p_title = vid.get('title', 'Unknown')
        v_frames_dir = vid["dir_path"]

        res = check_video_match(config = config, vlm_model = config['MODEL_NAME'], vlm_prompt = match_prompt, 
                                v_frames_dir = v_frames_dir, is_audio = is_audio, num_frames = vid['frame_num'], 
                                video_duration = vid['src_duration'], content = content, logger = logger)




        logger.info(f"===> Grounding: {p_title:<60} | Frame_id: {res.get('frame_id')}")

        frame_id = res.get('frame_id')

        if not isinstance(frame_id, int):
            logger.warning(f"[Grounding] N/A Frame_id for {p_title}: {frame_id}")
            return idx, None

        frame_id_str = "{:03d}".format(int(frame_id))
        v_frames_path = os.path.join(v_frames_dir, f"{frame_id_str}.jpg")
        evaluation_record = {
            "title": p_title,
            "path": v_frames_path,
            "vlm_result": res
        }
        return idx, evaluation_record

    N = len(candidates)
    all_evaluations = [None] * N 
    max_workers = min(4, N)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(evaluate_one_video, i, vid)
            for i, vid in enumerate(candidates)
        ]

        for future in as_completed(futures):
            try:
                idx, record = future.result()
                all_evaluations[idx] = record
            except Exception as e:
                logger.error(f"[Grounding] Failed: {e}")

    valid_results = [r for r in all_evaluations if r is not None]         

    with open(res_path, 'w', encoding='utf-8') as f:
        json.dump(valid_results, f, indent=4, ensure_ascii=False)
        
    logger.info(f"[Grounding] Result details are saved to {res_path}")


def get_gt_frame(config, data, frames_dir, logger):

    # Target Destination
    dst_path = os.path.join(frames_dir, 'ground_truth.jpg')

    if os.path.exists(dst_path):
        logger.info(f"[GT Frame] Target already exists, skipping: {dst_path}")
        return

    # Source Directory: defaults to './images' if not in config
    images_dir = config.get('GT_IMAGES_DIR', './images')
    vid_id = data['id']
    
    # Construct Source Path
    src_path = os.path.join(images_dir, f"{vid_id}.jpg")

    if os.path.exists(src_path):
        try:
            shutil.copy(src_path, dst_path)
            logger.info(f"[GT Frame] Copied GT frame from {src_path} to {dst_path}")
        except Exception as e:
            logger.error(f"[GT Frame] Failed to copy GT frame: {e}")
    else:
        logger.error(f"[GT Frame] Source image not found: {src_path}")

    
def extract_audio_clip(audio_path, out_audio_path, t, logger, duration=2.0):

    start = max(t - duration / 2, 0.0)

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-ss", f"{start:.3f}",
        "-t", f"{duration:.3f}",
        "-i", audio_path,
        "-ac", "1",
        "-ar", "16000",
        out_audio_path
    ]

    try:
        subprocess.run(cmd, check=True)
        logger.info(f"[AudioClip] Save audio clip to {audio_path}.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"[AudioClip] Failed: {out_audio_path}, {e}")
        return False


def extract_audio_test(is_audio, config, frames_dir, res_dir, vid, logger):

    if not is_audio:
        return

    logger.info(f">>>>>>>>>>>>>>>>>>>>> Extracting audiao >>>>>>>>>>>>>>>>>>>>>")

    f_path = os.path.join(frames_dir, 'ground_truth.jpg')

    if not os.path.exists(f_path):
        return


    audio_clip_dir = os.path.join(config['RESULTS_DIR'], "audio_clip", vid)

    os.makedirs(audio_clip_dir, exist_ok=True)


    result_json_path = os.path.join(res_dir, f"{vid}_result.json")

    with open(result_json_path, "r") as f:
        results = json.load(f)
    
    index_path = os.path.join(frames_dir, "frames_index.json")

    with open(index_path, 'r') as f: 
        candidates = json.load(f)
    

    def process_one(item, candidates, logger):

        frame_id = item["vlm_result"]["frame_id"]
        audio_dir = os.path.dirname(item["path"])

        candidate_info = next((x for x in candidates if x["dir_path"] == audio_dir), None)
        if candidate_info is None:
            logger.warning(f"[AudioClip] No candidate info for {audio_dir}")
            return item

        interval = candidate_info['src_duration'] / candidate_info['frame_num']
        audio_path = candidate_info["audio_path"]

        if not os.path.exists(audio_path):
            logger.warning(f"[AudioClip] Missing audio: {audio_path}")
            return item 

        t = (frame_id + 0.5) * interval
        title_clean = "".join([c for c in candidate_info['title'] if c.isalnum()])[:10]
        title_hash = hashlib.md5(candidate_info['title'].encode("utf-8")).hexdigest()[:4]
        out_audio = os.path.join(audio_clip_dir, f"{title_clean}_{title_hash}_{frame_id:03d}.wav")

        is_audio_clip = extract_audio_clip(audio_path=audio_path, out_audio_path=out_audio, t=t, logger=logger)
           
        if is_audio_clip:

            item["audio_path"] = out_audio

        return item


    max_workers = 2

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_one, item, candidates, logger) for item in results]

        new_results = []
        for future in as_completed(futures):
            try:
                new_results.append(future.result())
            except Exception as e:
                logger.error(f"[AudioClip] Worker failed: {e}")

    with open(result_json_path, "w") as f:
        json.dump(new_results, f, ensure_ascii=False, indent=2)

    logger.info(f"[AudioClip] Finished extracting {len(new_results)} audio clips")

    

def vlm_test(config, prompt, data, frames_dir, res_dir, vlm_test_dir, is_audio, content, logger):
    """
    VLM existence test:
    Images are loaded from *_result.json paths instead of frames_dir folders.
    """
    # ---- Load result json ----
    vid_id = data["id"]
    res_path = os.path.join(res_dir, f"{vid_id}_result.json")

    with open(res_path, "r", encoding="utf-8") as f:
        res_data = json.load(f)

    index_path = os.path.join(frames_dir, "frames_index.json")

    os.makedirs(vlm_test_dir, exist_ok=True)
    vlm_test_path = os.path.join(vlm_test_dir, f'{data["id"]}.json')

    with open(index_path, 'r') as f: 
        candidates = json.load(f)

    if not candidates:
        final_result = {
        "video_id": vid_id,
        "pass": False,
        "frame_details": []
        }
        with open(vlm_test_path, "w", encoding="utf-8") as f:
            json.dump(final_result, f, indent=4, ensure_ascii=False)
        logger.info(f"[VLM TEST] No results for {vid_id}")
        return

    f_path = os.path.join(frames_dir, 'ground_truth.jpg')

    if not os.path.exists(f_path):
        final_result = {
        "video_id": vid_id,
        "pass": False,
        "frame_details": []
        }
        with open(vlm_test_path, "w", encoding="utf-8") as f:
            json.dump(final_result, f, indent=4, ensure_ascii=False)
        logger.info(f"[VLM TEST] No results for {vid_id}")
        return

    # ---- Prepare prompt ----
    if not is_audio:
        prompt_for_test = prompt["TEST_PROMPT"].replace("$Text_Content$", str(content))
    else:
        prompt_for_test = prompt["AUDIO_TEST_PROMPT"].replace("$Text_Content$", str(content))

    # ---- Load Ground Truth Image ONCE ----
    ground_truth_path = os.path.join(frames_dir, "ground_truth.jpg")
    with open(ground_truth_path, "rb") as f:
        ground_truth_bytes = f.read()

    # ---- Load result json ----
    vid_id = data["id"]
    res_path = os.path.join(res_dir, f"{vid_id}_result.json")

    with open(res_path, "r", encoding="utf-8") as f:
        res_data = json.load(f)


    image_items = [
        {
            "title": item.get("title", ""),
            "path": item["path"],
            "frame_id": item.get("vlm_result", {}).get("frame_id")
        }
        for item in res_data
        if "path" in item
    ]

    logger.info(f"[VLM TEST] Loaded {len(image_items)} images from {res_path}")

    client = genai.Client(api_key=config["GEMINI_API_KEY"])

    MAX_RETRIES = 5
    BASE_DELAY = 4
    MAX_DELAY = 60

    video_pass = False
    frame_details = []

    for item in image_items:
        img_path = item["path"]

        api_success = False
        text_match = False
        image_match = False

        if not os.path.exists(img_path):
            continue

        with open(img_path, "rb") as f:
            image_bytes = f.read()

        parts=[types.Part(text=prompt_for_test),
                types.Part.from_bytes(data=ground_truth_bytes, mime_type="image/jpeg",),
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg",),
                ]


        if is_audio:
            audio_path = item.get("audio_path")
            if audio_path and os.path.exists(audio_path):
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()

                parts.append(
                    types.Part.from_bytes(
                        data=audio_bytes,
                        mime_type="audio/wav"
                    )
                )


        for attempt in range(MAX_RETRIES + 1):
            try:
                response = client.models.generate_content(
                    model=config["TEST_MODEL"],
                    contents=types.Content(
                        parts=parts
                    ),
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json",
                    ),
                )

                result_json = json.loads(response.text)
                text_match = bool(result_json.get("text_match", False))
                image_match = bool(result_json.get("image_match", False))

                api_success = True
                break

            except Exception as e:
                err = str(e)
                if "429" in err or "ResourceExhausted" in err or "Quota" in err:
                    if attempt < MAX_RETRIES:
                        sleep_time = min(
                            MAX_DELAY, BASE_DELAY * (2 ** attempt)
                        ) + random.uniform(0.5, 2.0)
                        logger.warning(
                            f"[VLM TEST] Rate limit on {img_path}, retry in {sleep_time:.1f}s"
                        )
                        time.sleep(sleep_time)
                        continue
                logger.error(f"[VLM TEST] API error on {img_path}: {e}")
                break

        if not api_success:
            continue

        logger.info(
            f"[VLM TEST] {img_path:<60} | Text: {text_match} | Image: {image_match}"
        )

        frame_details.append({
            "title": item["title"],
            "path": img_path,
            "text_match": text_match,
            "image_match": image_match
        })

        if text_match or image_match:
            video_pass = True

        time.sleep(0.5)

    # ---- Save result ----
    final_result = {
        "video_id": vid_id,
        "pass": video_pass,
        "frame_details": frame_details
    }

    with open(vlm_test_path, "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False)

    logger.info(f"[VLM TEST] Results saved to {vlm_test_path}")