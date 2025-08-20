from setuptools import setup, find_packages

setup(
    name="vantai-target-scoreboard",
    version="1.0.0",
    description="Modality-aware target scoring system built on Open Targets data",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "httpx>=0.25.0",
        "pydantic>=2.5.0",
        "pandas>=2.1.0",
        "networkx>=3.2",
        "streamlit>=1.28.0",
        "python-dotenv>=1.0.0",
        "plotly>=5.17.0",
        "requests>=2.31.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "ruff>=0.1.0"
        ]
    },
    python_requires=">=3.11",
    author="VantAI",
    author_email="engineering@vantai.com",
    license="Proprietary",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.11",
    ],
)
