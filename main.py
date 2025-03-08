import jieba
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class PaperChecker:
    def __init__(self, model_name="sentence-transformers/paraphrase-MiniLM-L6-v2", cache_dir="./models"):
        self.cache_dir = cache_dir
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 在初始化阶段加载模型
        self.model = SentenceTransformer(model_name, cache_folder=self.cache_dir, device=device)
        print(f"模型已加载至: {self.cache_dir}")

    def preprocess(self, text):
        """文本预处理（分句+去除空白）"""
        sentences = text.replace("\n", "。").split("。")  # 以句号拆分
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def embed_sentences(self, sentences):
        """将句子转换为向量"""
        return self.model.encode(sentences, convert_to_tensor=True)

    def check_similarity(self, text1, text2):
        """计算两篇论文的相似度"""
        sentences1 = self.preprocess(text1)
        sentences2 = self.preprocess(text2)

        embeddings1 = self.embed_sentences(sentences1)
        embeddings2 = self.embed_sentences(sentences2)

        # 计算余弦相似度
        similarity_matrix = cosine_similarity(embeddings1.cpu().numpy(), embeddings2.cpu().numpy())

        # 取最大相似度作为最终评分
        max_similarities = similarity_matrix.max(axis=1)
        overall_similarity = np.mean(max_similarities)
        return overall_similarity

# 示例
if __name__ == "__main__":
    paper1 = "人工智能是一门新兴技术。它已经应用在多个领域，比如医疗、金融和自动驾驶。"
    paper2 = "AI 是一种新技术，被广泛应用于医疗、金融和自动驾驶。"

    checker = PaperChecker()
    score = checker.check_similarity(paper1, paper2)
    print(f"论文相似度：{score:.4f}")
