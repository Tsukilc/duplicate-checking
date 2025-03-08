import jieba
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor

import os


class PaperChecker:
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 cache_dir="./models", use_gpu=True):
        self.cache_dir = cache_dir
        self.device = 'cuda' if torch.cuda.is_available() and use_gpu else 'cpu'

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        fast_model_path = os.path.join(self.cache_dir, "fast_model.pt")

        if os.path.exists(fast_model_path):
            print("加载优化后的模型...")
            self.model = torch.load(fast_model_path,
                                    weights_only=False,
                                    map_location=self.device)
        else:
            print("初始化新模型...")
            self.model = SentenceTransformer(model_name)
            # 保存为权重文件
            torch.save(self.model.state_dict(), fast_model_path)

        print(f"模型加载完成，设备: {self.device}")

    def preprocess(self, text):
        """文本预处理（分词+去除空白）"""
        sentences = text.replace("\n", "。").split("。")  # 以句号拆分
        sentences = [s.strip() for s in sentences if s.strip()]

        # 分词并打印每个句子的分词结果
        for sentence in sentences:
            words = jieba.cut(sentence)  # 分词
            word_list = ' '.join(words)  # 将分词结果拼接成字符串
            print(f"分词结果：{word_list}")  # 打印分词结果

        # 返回分词后的句子
        sentences = [' '.join(jieba.cut(sentence)) for sentence in sentences]
        return sentences

    def embed_sentences(self, sentences):
        """将句子转换为向量"""
        return self.model.encode(sentences, convert_to_tensor=True)

    def check_similarity(self, text1, text2, threshold=0.7):
        sentences1 = self.preprocess(text1)
        sentences2 = self.preprocess(text2)

        embeddings1 = self.embed_sentences(sentences1)
        embeddings2 = self.embed_sentences(sentences2)

        similarity_matrix = cosine_similarity(embeddings1.cpu().numpy(), embeddings2.cpu().numpy())
        max_similarities = similarity_matrix.max(axis=1)
        overall_similarity = np.mean(max_similarities)

        return (overall_similarity, "有重复") if overall_similarity >= threshold else (overall_similarity, "没有重复")

    def check_multiple_pairs(self, pairs, threshold=0.7):
        """并行检查多个文本对的相似度"""
        results = []
        with ThreadPoolExecutor() as executor:
            future_to_pair = {executor.submit(self.check_similarity, pair[0], pair[1], threshold): pair for pair in
                              pairs}
            for future in future_to_pair:
                similarity, result = future.result()
                results.append((similarity, result))
        return results


# 示例
if __name__ == "__main__":
    paper1 = "机器学习是人工智能的一个分支"
    paper2 = "深度学习属于机器学习。"

    # 初始化 PaperChecker
    checker = PaperChecker(model_name="sentence-transformers/paraphrase-xlm-r-multilingual-v1", cache_dir="./models",
                           use_gpu=True)

    # 比较两个句子的相似度
    score, result = checker.check_similarity(paper1, paper2, threshold=0.5)  # 设置阈值为 0.5
    print(f"论文相似度：{score:.4f}, 查重结果：{result}")
