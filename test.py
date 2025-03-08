import pickle

import pytest
import os
import tempfile

import torch

from main import PaperChecker


@pytest.fixture
def checker():
    """测试用查重器实例"""
    from main import PaperChecker
    return PaperChecker(use_gpu=False)


@pytest.fixture
def sample_texts():
    return {
        "empty": "",
        "short": "这是一个短文本。",
        "normal": "自然语言处理是人工智能的重要领域。深度学习模型在其中发挥关键作用。",
        "dup_part": "自然语言处理是AI的核心领域。深度学习模型非常重要。",
        "full_dup": "自然语言处理是人工智能的重要领域。深度学习模型在其中发挥关键作用。"
    }


# 测试工具函数
def create_temp_file(content, suffix=".txt"):
    """创建临时测试文件"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
        f.write(content)
        return f.name


# 测试预处理模块
class TestPreprocess:
    def test_jieba_segment(self, checker):
        text = "自然语言处理技术"
        result = checker.preprocess(text)
        assert result == ["自然语言 处理 技术"]

class TestModelLoading:
    @pytest.fixture(autouse=True)
    def clean_cache(self):
        # 清理模型缓存
        cache_path = os.path.join("./models", "fast_model.pt")
        if os.path.exists(cache_path):
            os.remove(cache_path)
        yield

    def test_cold_start(self):
        """测试无缓存时的模型加载"""
        checker = PaperChecker(use_gpu=False)
        assert checker.model is not None

    def test_warm_start(self):
        """测试有缓存时的模型加载"""
        # 先冷启动生成缓存
        PaperChecker(use_gpu=False)
        # 再次加载
        checker = PaperChecker(use_gpu=False)
        assert checker.model is not None


# 测试文件操作
class TestFileOperations:
    def test_valid_file(self):
        from main import validate_file_path
        valid_file = create_temp_file("正常内容" * 100)
        assert validate_file_path(valid_file) == os.path.abspath(valid_file)

    def test_invalid_paths(self):
        from main import validate_file_path
        with pytest.raises(FileNotFoundError):
            validate_file_path("non_existent.txt")

        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(IsADirectoryError):
                validate_file_path(d)

    def test_file_size_limit(self):
        from main import validate_file_path
        big_file = create_temp_file("a" * (10 * 1024 * 1024 + 1))
        with pytest.raises(ValueError):
            validate_file_path(big_file)


# 测试完整流程
class TestIntegration:
    def test_full_workflow(self, checker, tmp_path):
        """测试完整流程并验证输出文件"""
        from main import compare_papers

        # 使用临时目录
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        os.chdir(output_dir)

        # 生成测试内容
        orig_content = "自然语言处理的基本原理。"
        plag_content = "自然语言处理的基础理论。"

        # 执行查重比较
        results = compare_papers(
            orig_text=orig_content,
            plagiarism_texts=[("test.txt", plag_content)],  # 文件名不影响测试
            checker=checker,
            threshold=0.5
        )



# 异常处理测试
class TestExceptionHandling:
    def test_invalid_encoding(self):
        """测试读取非UTF-8编码文件"""
        from main import read_file

        # 生成包含非法字节的文件
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".txt") as f:
            f.write(b"\x80\xFFinvalid")  # 直接写入二进制非法数据
            bad_file = f.name

        try:
            with pytest.raises(UnicodeDecodeError):
                read_file(bad_file)
        finally:
            os.unlink(bad_file)


# 清理临时文件
@pytest.fixture(autouse=True)
def cleanup():
    yield
    for f in os.listdir(tempfile.gettempdir()):
        if f.startswith("tmp"):
            try:
                os.remove(os.path.join(tempfile.gettempdir(), f))
            except:
                pass