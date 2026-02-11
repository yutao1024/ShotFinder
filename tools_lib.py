import os
import re
import json
import yt_dlp
import logging
import datetime
import yaml
import hashlib
import http.client
import concurrent.futures
import warnings



class GeminiWarningFilter(logging.Filter):
    def filter(self, record):
        if "thought_signature" in record.getMessage():
            return False
        if "AFC is enabled" in record.getMessage():
            return False
        if "HTTP Request: POST" in record.getMessage():
            return False
        return True

def create_logger(log_path, logger_name=None):

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger_name = logger_name or f"logger_{os.path.basename(log_path)}"
    logger = logging.getLogger(logger_name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        def converter(sec, what=None):
            utc_dt = datetime.datetime.fromtimestamp(sec, datetime.timezone.utc)
            dt = utc_dt + datetime.timedelta(hours=8)
            return dt.timetuple()

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        formatter.converter = converter

        fh = logging.FileHandler(log_path, encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        if 'GeminiWarningFilter' in globals():
            f = GeminiWarningFilter()
            logger.addFilter(f)
            fh.addFilter(f)
            sh.addFilter(f)

    return logger

def read_yaml(file_path):
    """
    Reads a single YAML file and returns a dictionary.
    """
    try:
        # Open file with utf-8 encoding
        with open(file_path, 'r', encoding='utf-8') as file:
            # Use safe_load to convert YAML to a Python dictionary
            return yaml.safe_load(file)
            
    except FileNotFoundError:
        print(f"Error: File not found {file_path}")
        return {}
    except yaml.YAMLError as exc:
        print(f"Error: Failed to parse YAML - {exc}")
        return {}

def parse_timestamp(t_str):
    try:
        parts = [float(x) for x in t_str.strip().split(':')]
        if len(parts) == 3: return parts[0]*3600 + parts[1]*60 + parts[2]
        if len(parts) == 2: return parts[0]*60 + parts[1]
        return float(t_str)
    except: return None


def identify_platform(url):
    if "youtube.com" in url or "youtu.be" in url: 
        return "YouTube"
    return "Unknown"

def extract_and_join(data_json, keys):
    parts = []
    for key in keys:
        value = data_json.get(key)
        if isinstance(value, list): 
            list_content = " ".join(str(item).strip() for item in value if str(item).strip())
            if list_content:
                parts.append(list_content)
        elif isinstance(value, str): 
            value = value.strip()
            if value:
                parts.append(value)
    return " ".join(parts)


def format_user_input(data_json):

    parts = []
    description = data_json.get('Description')
    clip_description = data_json.get('Clip Description')

    audio = False
    resolution = False

    if not description:
        return clip_description, audio, resolution

    if description == "Resolution":
        resolution = True
        return clip_description, audio, resolution

    key = "{} Description".format(description)

    parts = ['Clip Description', key]

    content = extract_and_join(data_json, parts)

    if description in ["Temporal", "Color", "Style"]:

        return content, audio, resolution

    if description == "Audio":

        audio = True
        return content, audio, resolution
    

def has_720p_format(info):
    formats = info.get('formats', [])
    for f in formats:
        h = f.get('height')
        w = f.get('width')

        if h == 720 or w == 720:
            return True
            
    return False

def has_1080p_format(info):
    formats = info.get('formats', [])
    for f in formats:
        h = f.get('height')
        w = f.get('width')

        if h == 1080 or w == 1080:
            return True
            
    return False


def check_youtube_video(config, url, logger, is_resolution, check_duration=True, check_format=True):

    cookies_path = config['COOKIES_FILE']
    max_sec = config['MAX_SEC']

    ydl_opts = {
        'quiet': True,
        'ignoreerrors': True,
        'noplaylist': True,
        'skip_download': True,
        'nocheckcertificate': True,
        'cookiefile': cookies_path,
        "--retries": "5",
        'http_headers': {'User-Agent': 'Mozilla/5.0 ...'},
        'js_runtimes': {'node': {'path': config['NODE_PATH']}},
        'remote_components': ['ejs:github'],
    }

    if logger:
        ydl_opts['logger'] = logger

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)

        if not info:
            return False
        
        if not (info.get('id') or info.get('title') or info.get('is_live', False)):
            return False

        if check_duration and max_sec is not None:
            duration = info.get('duration')
            if duration is None or duration > max_sec:
                return False

        if check_format and not is_resolution:
            if not has_720p_format(info):
                return False
            
        if check_format and is_resolution:
            if not has_1080p_format(info):
                return False

        return True

    except yt_dlp.utils.DownloadError as e:
        if logger:
            logger.error(f"DownloadError: {e} | Skipping {url}")
        return False
    except Exception as e:
        if logger:
            logger.error(f"Unexpected error: {e} | Skipping {url}")
        return False


def search_videos(config, queries, serpapi_key, is_resolution, logger):

    all_candidates = []

    for i, query in enumerate(queries):

        query = re.sub(r'[^\w\s]', '', query)
        final_query = f"{query} youtube" 
        logger.info(f"[Tool] Searching {i+1}/{len(queries)} keyword: {final_query}")

        candidates = []
        page = 1
        max_results = config['URL_NUM']

        max_page = config['MAX_PAGE']

        try:

            conn = http.client.HTTPSConnection("google.serper.dev")

            headers = {
                'X-API-KEY': serpapi_key,
                'Content-Type': 'application/json'
            }
            
            while len(candidates) < max_results and page <= max_page:

                payload = json.dumps({
                    "q": final_query,
                    "num": 10,
                    "page": page,
                    "hl": "en",  
                    "gl": "us"   
                })

                logger.info(f"[Tool] Searching page: {page}")

                conn.request("POST", "/search", payload, headers)

                res = conn.getresponse()
                data = res.read()
                results = json.loads(data.decode("utf-8"))
        
                organic_results = results.get("organic", results.get("organic_results", []))

                if not organic_results:
                    logger.info("[Tool] No more organic results, stop paging.")
                    break

                for res in organic_results:
                    link = res.get("link", "")
                    platform = identify_platform(link)

                    if platform == "YouTube" and ("watch?v=" in link or "shorts" in link) and check_youtube_video(config, link, logger, is_resolution):
                        if link not in candidates and link not in all_candidates:
                            candidates.append(link)
                
                page += 1

            if not candidates:
                logger.warning("[Tool] No video results found.")
                all_candidates.extend([]) 
                
            logger.info('=============================== Search Results ====================================')
            for item in candidates:
                logger.info(item) 
            logger.info('===================================================================================')

            all_candidates.extend(candidates[:max_results])
            
        except Exception as e:
            logger.error(f"[Tool] Search error: {e}")
            all_candidates.extend([])

    return all_candidates

def get_unique_filename(info, save_dir, platform="Unknown", prefix="video"):

    vid_id = info.get("id")
    url = info.get("webpage_url", "")
    hash_str = hashlib.md5(url.encode()).hexdigest()[:6]
    filename = f"{save_dir}/{prefix}_{platform}_{vid_id}_{hash_str}.mp4"
    return filename


def download_single(url, ydl_opts, save_dir, logger, platform="Unknown", prefix="video"):

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if not info:
                logger.error(f"[Downloading] Failed to extract info: {url}")
                return None

            final_path = get_unique_filename(info, save_dir, platform, prefix)


            temp_path = ydl.prepare_filename(info)
            base, _ = os.path.splitext(temp_path)
            temp_path = base + ".mp4"

            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                logger.error(f"[Downloading] File missing or empty: {temp_path}")
                return None

            os.replace(temp_path, final_path)
            logger.info(f"[Downloading] Success: {info.get('title')} -> {final_path}")

            return {
                "status": "success",
                "path": final_path,
                "url": url,
                "title": info.get("title"),
                "platform": platform
            }

    except Exception as e:
        logger.error(f"[Downloading] Failed: {url} - {e}")
        return None

def download_ground_truth(config, url, ydl_opts, save_dir, cookies_path,logger):

    ydl_opts = {
        'format': "bestvideo+bestaudio/best",
        'merge_output_format': 'mp4',
        'outtmpl': f'{save_dir}/%(id)s.%(ext)s',
        'noplaylist': True,
        'quiet': True,
        'ignoreerrors': True,
        'cookiefile': cookies_path,
        'geo_bypass': True,
        'socket_timeout': 180,
        "--extractor-retries": "5", 
        "--retries": "5",
        'js_runtimes': {'node': {'path': config['NODE_PATH'],}},
        'remote_components': ['ejs:github'],
    }
    return download_single(url, ydl_opts, save_dir, logger, platform="GroundTruth", prefix="gt")


def download_videos_deterministic(config, url_list, save_dir, cookies_path, is_resolution, logger):

    if not url_list:
        logger.info("[Downloading] No URLs provided.")
        return []

    logger.info(f"[Downloading] Starting batch download of {len(url_list)} videos (including GT video)")

    if is_resolution:
        format_str = (
            'bestvideo[height=1080]+bestaudio/'
            'best[height=1080]'
        )
        logger.info(f"[Downloading] ===> Resolution is set to 1080P.")
    else:  
        format_str = (
            'bestvideo[height=720]+bestaudio[ext=m4a]/'
            'best[ext=mp4][height=720]'
        )

    ydl_opts = {
        'format': format_str,
        'merge_output_format': 'mp4',
        'outtmpl': f'{save_dir}/%(id)s.%(ext)s',
        'noplaylist': True,
        'quiet': True,
        'ignoreerrors': True,
        'cookiefile': cookies_path,
        'geo_bypass': True,
        'socket_timeout': 180,
        "--extractor-retries": "5", 
        "--retries": "5",
        'js_runtimes': {'node': {'path': config['NODE_PATH'],}},
        'remote_components': ['ejs:github'],
    }

    downloaded_meta = []


    if len(url_list) == 1:

        gt_meta = download_ground_truth(config, url_list[0], ydl_opts, save_dir, cookies_path, logger)
        if gt_meta:
            logger.info("[Downloading] Ground Truth Video Downloaded.")
        return downloaded_meta

    candidate_urls = url_list[:-1]
    gt_url = url_list[-1]

    max_workers = min(len(candidate_urls), 4) 
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(download_single, url, ydl_opts, save_dir, logger, platform="YouTube")
            for url in candidate_urls
        ]

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                downloaded_meta.append(result)

    gt_meta = download_ground_truth(config, gt_url, ydl_opts, save_dir, cookies_path, logger)
    is_gt_video = False
    if gt_meta:
        is_gt_video = True
        logger.info("[Downloading] Ground Truth Video Downloaded.")
    return downloaded_meta, is_gt_video








