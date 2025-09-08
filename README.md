# Autocode

Research on technology that automatically generates high-quality source code from requirements written in natural language, execution examples, or partially written source code.

>-  Automatic source code generation technology that combines new and existing techniques such as machine learning (language model), program synthesis, and software engineering.

![image](./overview_autocode.png)

## Overview

This repository contains tools and frameworks for code generation and evaluation using language models. The project leverages LangGraph and various language model APIs to provide a robust environment for code generation tasks.

## Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (recommended for optimal performance)
- Virtual environment (recommended)

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv test
source test/bin/activate 
```

2. Install dependencies:
```bash
pip install -r requirements.txt
git submodule update --init --recursive
pip install -e third_party/etri_langgraph
```

## Configuration

1. Create an `api_keys.json` file in the root directory with the following structure:
```json
{
  "OPEN_WEBUI_BASE_URL": "your-model-url",
  "OPENAI_API_KEY": "your-api-key"
}
```

## Usage

The project provides two main functionalities:

### Code Generation

#### LLaMA-3
```bash
python3 run.py generator \
    --config_path=configs/llama3_test.yaml \
    - run \
    - merge_json \
    - exit
```

#### GPT-4
```bash
python3 run.py generator \
    --config_path=configs/gpt4_test.yaml \
    - run \
    - merge_json \
    - exit
```

## Project Structure

```
autocode/
├── configs/         # Configuration files
├── templates/       # Template files
├── third_party/     # Third-party dependencies
├── .gitmodules      # Submodule information
├── api_keys.json    # API key configuration
├── run.py           # Main entry point
└── requirements.txt # Python dependencies
```

## Development

- Configuration files are stored in `configs/`
- Templates for code generation are in `templates/`

