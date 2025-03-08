import pytest
import os
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock

# 测试基础配置
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, "test_data")


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
    def test_full_workflow(self, checker):
        from main import read_file, compare_papers

        orig_content = "自然语言处理的基本原理。"
        plag_content = "自然语言处理的基础理论。"

        orig_file = create_temp_file(orig_content)
        plag_file = create_temp_file(plag_content)

        result = compare_papers(
            orig_content,
            [(plag_file, plag_content)],
            checker
        )

        assert "相似度" in result[0]
        assert os.path.exists("ans.txt")



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