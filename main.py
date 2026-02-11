import os
import re
import json
import warnings
import argparse
import datetime
from zoneinfo import ZoneInfo
from tools_lib import create_logger, read_yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from run_search_grounding import run_search, download_videos, process_videos, run_grounding, get_gt_frame, extract_audio_test, vlm_test

warnings.filterwarnings("ignore", message="Using a slow image processor as `use_fast` is unset.*")
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Process video search and grounding.')
parser.add_argument('--config_path', type=str, default='./config/config.yaml', help='Path to configuration file')
parser.add_argument('--prompt_path', type=str, default='./config/prompt.yaml', help='Path to prompt file')
parser.add_argument('--input_path', type=str, required=True, help='Path to the input JSON file containing video data')
parser.add_argument('--model_name', type=str, default=None, help='Model Name for Search and Grounding.')



args = parser.parse_args()


download_lock = Lock()

def process_data_with_download_lock(data, config, prompt, task_log_dir):
    data_id = data["id"]
    data_log_path = os.path.join(task_log_dir, f"{data_id}.log")

    logger = create_logger(data_log_path, logger_name=f"data_{data['id']}")

    logger.info(f"================ Processing data {data_id} ================")
    logger.info(f"Using Model: {config.get('MODEL_NAME')}")

    # 1. Search
    target_urls, is_audio, is_resolution, content = run_search(config=config, data=data, search_prompt=prompt['SEARCH_PROMPT'], logger=logger)
    
    # 2. Download
    with download_lock:
        download_videos(config = config, data = data, target_urls = target_urls, is_resolution = is_resolution, logger = logger)

    frames_dir = os.path.join(config['EXTRACT_FRAMES'], data["id"])
    os.makedirs(frames_dir, exist_ok=True)

    # 3. Process Videos (Extract frames)
    process_videos(frames_dir=frames_dir, vid=data["id"], config=config, is_audio=is_audio, logger=logger)


    # 4. Run Grounding
    res_dir = os.path.join(config['RESULTS_DIR'], "grounding_result")
    os.makedirs(res_dir, exist_ok=True)

    run_grounding(config = config, frames_dir = frames_dir, res_dir=res_dir,data = data, prompt = prompt, is_audio = is_audio, content = content, logger = logger)

    # 5. Get GT Frame
    get_gt_frame(config=config, data=data, frames_dir=frames_dir, logger=logger)

    # 6. Audio Extract Test
    extract_audio_test(is_audio=is_audio, config = config, frames_dir= frames_dir, res_dir=res_dir, vid = data['id'], logger=logger)

    # 7. VLM Test
    vlm_test_dir = os.path.join(config['RESULTS_DIR'], "vlm_test")
    os.makedirs(vlm_test_dir, exist_ok=True)
    vlm_test(config=config, prompt=prompt, data=data, frames_dir=frames_dir,  res_dir=res_dir, vlm_test_dir=vlm_test_dir, is_audio=is_audio, content=content, logger=logger)

    logger.info(f"================ Finished data {data_id} ================")

    return True


def main():
    config = read_yaml(args.config_path)
    prompt = read_yaml(args.prompt_path)

    # === Modification 3: Enforce unified Search and VLM model ===
    if args.model_name:
        print(f"Setting Model to: {args.model_name}")
        config['MODEL_NAME'] = args.model_name

    # === Modification 4: Directly load the JSON file from the specified path ===
    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input file not found: {args.input_path}")

    with open(args.input_path, 'r', encoding='utf-8') as json_f:
        datas = json.load(json_f)

    input_filename = os.path.basename(args.input_path)

    # === Create directories ===
    for key in ['OUTPUT_DIR', 'LOG_DIR', 'EXTRACT_FRAMES', 'RESULTS_DIR']:
        os.makedirs(config[key], exist_ok=True)

    time_now = datetime.datetime.now(ZoneInfo("Asia/Shanghai")).strftime('%Y-%m-%d-%H-%M-%S')
    # Log directory includes input filename for easy distinction
    task_log_dir = os.path.join(config['LOG_DIR'], f"task_{input_filename}_{time_now}")
    os.makedirs(task_log_dir, exist_ok=True)

    # === Record configuration ===
    config_log_path = os.path.join(task_log_dir, "config_and_prompt.log")
    with open(config_log_path, "w", encoding="utf-8") as log_f:
        log_f.write("=====================================CONFIG========================================\n")
        for key_, value_ in config.items():
            log_f.write(f"{key_}: {value_}\n")
        log_f.write("=====================================PROMPT========================================\n")
        log_f.write(f"PROMPT: {prompt['SEARCH_PROMPT']}\n")
        log_f.write("===================================================================================\n")
        log_f.write(f">>>>>>>>>>>>>>>>>>>> Processing {len(datas)} videos from {args.input_path} >>>>>>>>>>>>>>>>>>>> \n")

    # === Execute tasks ===
    max_workers = min(8, len(datas))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_data_with_download_lock, data, config, prompt, task_log_dir)
            for data in datas
        ]
        for f in as_completed(futures):
            f.result() 

    with open(config_log_path, "a", encoding="utf-8") as log_f:
        log_f.write(f"[{time_now}] All {len(datas)} tasks finished. Logs are in {task_log_dir}\n")


if __name__ == '__main__':
    main()