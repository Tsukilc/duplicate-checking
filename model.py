from sentence_transformers import SentenceTransformer, util

MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
CACHE_DIR = "./models"

# 下载并加载模型（仅首次联网下载，以后可离线使用）
model = SentenceTransformer(MODEL_NAME, cache_folder=CACHE_DIR)

# 示例：文本转换为向量
sentence = ["今天周五我要吃麦当劳的西红柿", "今天星期五我要吃肯德基的番茄"]
embeddings = model.encode(sentence, convert_to_tensor=True)

# 计算语义相似度
similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
print(f"相似度: {similarity.item():.4f}")
