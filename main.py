import jieba
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import model


class PaperChecker:
    def __init__(self, model_name=model.MODEL_NAME, cache_dir="./models"):
        self.cache_dir = cache_dir
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name, cache_folder=self.cache_dir, device=device)
        print(f"模型已加载至: {self.cache_dir}")

    def preprocess(self, text):
        """文本预处理（分词+去除空白）"""
        sentences = text.replace("\n", "。").split("。")  # 以句号拆分
        sentences = [s.strip() for s in sentences if s.strip()]

        # 返回分词后的句子
        sentences = [' '.join(jieba.cut(sentence)) for sentence in sentences]
        return sentences

    def embed_sentences(self, sentences):
        """将句子转换为向量"""
        return self.model.encode(sentences, convert_to_tensor=True)

    def check_similarity(self, text1, text2, threshold=0.7):
        """计算两篇论文的相似度，并根据阈值判断是否为重复"""
        sentences1 = self.preprocess(text1)
        sentences2 = self.preprocess(text2)

        embeddings1 = self.embed_sentences(sentences1)
        embeddings2 = self.embed_sentences(sentences2)

        # 计算余弦相似度
        similarity_matrix = cosine_similarity(embeddings1.cpu().numpy(), embeddings2.cpu().numpy())

        # 取最大相似度作为最终评分
        max_similarities = similarity_matrix.max(axis=1)
        overall_similarity = np.mean(max_similarities)

        # 根据阈值判断是否为重复
        if overall_similarity >= threshold:
            return overall_similarity, "有重复"
        else:
            return overall_similarity, "没有重复"


# 示例
if __name__ == "__main__":
    paper1 = "我爱吃番茄，也很爱吃西红柿"
    paper2 = "我喜欢吃西红柿和番茄。"

    checker = PaperChecker(model_name=model.MODEL_NAME, cache_dir="./models")
    score, result = checker.check_similarity(paper1, paper2, threshold=0.5)  # 设置阈值为 0.5
    print(f"论文相似度：{score:.4f}, 查重结果：{result}")

