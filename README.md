# ShotFinder: Imagination-Driven Open-Domain Video Shot Retrieval via Web Search


<p align="center">
  <img src="figures/title.png" width="70%">
</p>

<font size=3><div align='center' >  
[[📝 arXiv Paper](https://arxiv.org/abs/2601.23232)] [[🗂️ ShotFinder Data](https://huggingface.co/datasets/Kirito-Lab/ShotFinder)] 
</div></font>

---

### 📑 Table of Contents

- [ 🏷️ Key Components](#key-components)
- [ ✨ Resources](#-resources)
- [ 🚀 Quick Start](#-quick-start)
- [ 📜 Citation](#cite)

---

<a id="key-components"></a>
### 🏷️ Key Components


#### 1. **ShotFinder Benchmark**

<p align="center">
  <img src="figures/dataset.png" width="100%">
</p>

- **Curated Open-Domain Collection**: Contains **1,210 high-quality video samples** collected from YouTube across **20 diverse thematic categories** (e.g., Knowledge, Gaming, Fashion, Documentaries).

- **Constraint-Driven Task Design**: Defines 6 specific task settings: a core **Shot Description** task plus **5 single-factor constraints** (Temporal order, Color, Visual style, Audio, and Resolution) to isolate and analyze specific retrieval capabilities.

- **Human-Verified Construction**: Utilizes a **Model-based Description Generation** pipeline (using Gemini-3-Pro) followed by rigorous **Human Verification and Refinement** to ensure the dataset is semantically accurate and free of noise.

- **Constraint-Aware Topic Allocation**: Implements a strategic mapping of topics to constraints (e.g., mapping "Music" to Style tasks or "Fitness" to Temporal tasks) to reduce content bias while ensuring relevance.


#### 2. **ShotFinder Method**

<p align="center">
  <img src="figures/pipeline.png" width="100%">
</p>

- **Three-Stage Retrieval Framework**: A text-driven system designed to locate specific shots in open-domain videos through **Generator**, **Retriever**, and **Localizer** modules.

- **Stage 1: Query Expansion via Video Imagination**: Unlike simple keyword extraction, the **Generator** uses an VLM to "imagine" the full video content and title from a short shot description, bridging the semantic gap between a specific shot and searchable video metadata.

- **Stage 2: Web Video Retrieval**: The **Retriever** interacts with search engines using the imagined queries to filter and download a set of candidate videos from the web.

- **Stage 3: Description-Guided Temporal Localization**: The **Localizer** employs MLLMs with **adaptive frame sampling** to analyze the candidate videos and pinpoint the exact start and end times of the target shot based on the text description.


---



### ✨ Resources
- 📝 **Paper (arXiv):**  https://arxiv.org/abs/2601.23232
- 🗂️ **Dataset (HF):**  https://huggingface.co/datasets/Kirito-Lab/ShotFinder

---

### 🚀 Quick Start


#### 1. **Clone this repository and navigate to the folder:**
```bash
git clone https://github.com/yutao1024/ShotFinder.git
cd ShotFinder
```

#### 2. **Install the inference package:**
```bash
conda create -n shotfinder python=3.11.5 -y
conda activate shotfinder
pip install -r requirements.txt
```

#### 3. **Data Preparation:**
Download [🗂️ ShotFinder Data](https://huggingface.co/datasets/Kirito-Lab/ShotFinder), unzip the datasets. The final structure should look like this:
```
ShotFinder/
├── data_json/               # Input Data
│   └── video_dataset.json   # Source file containing video metadata
├── images/                  # Ground Truth Images
│   ├── youtube_hE3tW0ujXEM.jpg             # Matches video ID "youtube_hE3tW0ujXEM"
│   ├── youtube_Oa0ZHfcalCM.jpg             # Matches video ID "youtube_Oa0ZHfcalCM"
│   └── ...
```

#### 4. **Configuration:**

Before running the code, you need to configure your **API** keys and environment settings.

1. **API Keys**: Open `config/config.yaml` (create it if it doesn't exist, using the structure below) and add your API keys. You will need a **SerpApi** key for the search agent and VLM keys for the models you intend to use.
```yaml
# config/config.yaml example
GEMINI_API_KEY: "your_gemini_key"
OPENAI_API_KEY: "your_openai_key"
ANTHROPIC_API_KEY: "your_anthropic_key"
SERPAPI_KEY: "your_serpapi_key" # Required for video search

# Paths
OUTPUT_DIR: "./output"
LOG_DIR: "./logs"
EXTRACT_FRAMES: "./frames"
RESULTS_DIR: "./results"
GT_IMAGES_DIR: "./images"
COOKIES_FILE: "./config/cookies.txt" # Required for yt-dlp
NODE_PATH: "/usr/bin/node" # Path to node.js (required for some yt-dlp extractors)

# Search Settings
URL_NUM: 2 # Number of chosen URLs
MAX_PAGE: 5 # Web pages to search
MAX_SEC: 3600 # Max video duration in seconds
```

2. **YouTube Cookies**: To ensure stable video downloading with `yt-dlp`, export your YouTube cookies to a Netscape formatted text file (e.g., using a browser extension) and save it as `config/cookies.txt`.
3. **Dependencies**: Ensure `ffmpeg` and `node` are installed on your system, as they are required for video processing and downloading.
```bash
sudo apt update
sudo apt install -y ffmpeg nodejs npm
```

4. **Prompt Configuration**: Modify `config/prompt.yaml` to adjust the `SEARCH_PROMPT`. Specifically, you should update the instruction regarding the quantity of keywords to control the number of search queries generated per attempt, and adjust the quantity of the provided examples in the `=== EXAMPLES ===` section to ensure consistency with your requirements.




#### 5. **Run Inference**

Use `main.py` to run the full pipeline: Input -> LLM Query -> Search Engine Interaction -> MLLM Grounding -> Output.

```bash
python main.py \
  --input_path ./data_json/video_dataset.json \
  --model_name "gemini-3-pro" \
  --config_path ./config/config.yaml
```

**Parameters:**

* `--input_path`: Path to the input JSON dataset.
* `--model_name`: The name of the VLM to use (`gemini-3-pro`,`gemini-2.5-pro`, `gpt-5.2`, `gpt-5-mini`, `claude-4.0-Sonnet`, `qwen3-omni-30b-a3b`, `qwen3-vl-235b-a22b`).
* `--config_path`: Path to your configuration file (default: `./config/config.yaml`).


#### 6. **Calculate the accuracy**

After running the inference, you can calculate the accuracy using `evalution.py`. This script compares the grounding results against the dataset annotations and calculates accuracy across different categories (`Shot`, `Temporal`, `Color`, `Style`, `Resolution`, `Audio` and `Avg.`).

```bash
python evalution.py \
  --model_name "gemini-2.5-Pro" \
  --dataset_path "./data_json/video_dataset.json" \
  --result_dir "./results/grounding_result" \
  --output_file "./results/evaluation_report.txt"
```

---


<a id="cite"></a>
### 📜 Citation


If you find it useful for your research and applications, please cite related papers/blogs using this BibTeX:
```bibtex
@misc{yu2026shotfinderimaginationdrivenopendomainvideo,
      title={ShotFinder: Imagination-Driven Open-Domain Video Shot Retrieval via Web Search}, 
      author={Tao Yu and Haopeng Jin and Hao Wang and Shenghua Chai and Yujia Yang and Junhao Gong and Jiaming Guo and Minghui Zhang and Xinlong Chen and Zhenghao Zhang and Yuxuan Zhou and Yufei Xiong and Shanbin Zhang and Jiabing Yang and Hongzhu Yi and Xinming Wang and Cheng Zhong and Xiao Ma and Zhang Zhang and Yan Huang and Liang Wang},
      year={2026},
      eprint={2601.23232},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.23232}, 
}
```


