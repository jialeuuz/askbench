# mind-eval/setup.py
from setuptools import setup, find_packages

setup(
    name="mind_eval",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "requests",
        "aiohttp",
        "tqdm",
        "numpy",
        "sympy",
        "absl-py",
        "langdetect",
        "immutabledict",
        "nltk",
        "timeout_decorator",
        "datasets",
        "scipy",
        "word2number",
        "latex2sympy2==1.9.1",
        "accelerate",
        "jsonlines"
        # 其他依赖...
    ],
)