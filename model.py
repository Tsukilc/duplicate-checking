from sentence_transformers import SentenceTransformer, util

MODEL_NAME = "paraphrase-xlm-r-multilingual-v1"
CACHE_DIR = "./models"

# 下载并加载模型（仅首次联网下载，以后可离线使用）
model = SentenceTransformer(MODEL_NAME, cache_folder=CACHE_DIR)

