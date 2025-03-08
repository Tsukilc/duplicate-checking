import re
from functools import partial

import jieba
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import os


class PaperChecker:
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 cache_dir="./models", use_gpu=True):
        self.cache_dir = cache_dir
        self.device = 'cuda' if torch.cuda.is_available() and use_gpu else 'cpu'

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        fast_model_path = os.path.join(self.cache_dir, "fast_model.pt")
        torch.set_num_threads(12)

        if os.path.exists(fast_model_path):
            # print("加载优化后的模型...")
            self.model = torch.load(fast_model_path,
                                    weights_only=False,
                                    map_location=self.device)
        else:
            print("初始化新模型...")
            self.model = SentenceTransformer(model_name)
            # 保存为权重文件
            torch.save(self.model.state_dict(), fast_model_path)


        # print(f"模型加载完成，设备: {self.device}")

    def preprocess(self, text):
        """文本预处理（分词+去除空白）"""
        sentences = text.replace("\n", "。").split("。")  # 以句号拆分
        sentences = [s.strip() for s in sentences if s.strip()]

        # # 分词并打印每个句子的分词结果
        # for sentence in sentences:
        #     words = jieba.cut(sentence)  # 分词
        #     word_list = ' '.join(words)  # 将分词结果拼接成字符串
        #     print(f"分词结果：{word_list}")  # 打印分词结果

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

def read_file(file_path):
    """读取文件内容"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def compare_papers(orig_text, plagiarism_texts, checker, threshold=0.5):
    results = []

    def _process_single(file_name, paper_text):
        """单任务处理函数"""
        score, result = checker.check_similarity(orig_text, paper_text, threshold)
        return f"与论文 {file_name} 的相似度：{score:.4f}, 查重结果：{result}\n"

    with ThreadPoolExecutor(max_workers=8) as executor:
        # 提交所有任务（保持顺序）
        futures = [
            executor.submit(_process_single, file_name, paper_text)
            for file_name, paper_text in plagiarism_texts
        ]

        # 按提交顺序获取结果
        results = [future.result() for future in futures]

    return results

def main():
    # 用户输入原始文件路径
    orig_file_path = input("请输入原始论文的路径（例如: test\\orig.txt）: ")

    # 启动异步加载模型
    with ThreadPoolExecutor(max_workers=2) as executor:
        # 提交模型加载任务
        checker_future = executor.submit(
            PaperChecker,
            model_name="sentence-transformers/paraphrase-xlm-r-multilingual-v1",
            cache_dir="./models",
            use_gpu=True
        )

        # 用户输入抄袭论文路径（此时模型加载在后台进行）
        plagiarism_file_path = input("请输入要对比的抄袭论文路径（例如: test\\orig_0.8_add.txt）: ")

        # 等待模型加载完成
        checker = checker_future.result()

        # 异步读取文件内容（直接传递原始文本）
        orig_text_future = executor.submit(read_file, orig_file_path)
        plagiarism_text_future = executor.submit(read_file, plagiarism_file_path)

        # 获取原始文本内容
        orig_text = orig_text_future.result()
        plagiarism_text = plagiarism_text_future.result()

        # 直接调用检查相似度方法
        score, result = checker.check_similarity(orig_text, plagiarism_text)

        # 输出结果到文件
        with open("ans.txt", "w", encoding="utf-8") as f:
            f.write(f"原文: {orig_file_path}\n")
            f.write(f"抄袭论文: {plagiarism_file_path}\n\n")
            f.write("查重结果：\n\n")
            f.write(f"与论文 {plagiarism_file_path} 的相似度：{score:.4f}, 查重结果：{result}\n")

        print("查重结果已输出到 ans.txt")

if __name__ == "__main__":
    main()