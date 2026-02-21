from setuptools import setup, find_packages

setup(
    name="caregiving-lm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "langchain-openai",
        "langchain-chroma",
        "chromadb",
        "openai",
        "python-dotenv",
        "numpy",
    ],
    python_requires=">=3.8",
) 