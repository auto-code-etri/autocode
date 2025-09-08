# Autocode

Research on technology that automatically generates high-quality source code from requirements written in natural language, execution examples, or partially written source code.

>-  Automatic source code generation technology that combines new and existing techniques such as machine learning (language model), program synthesis, and software engineering.

![image](./overview_autocode.png)

## Overview

This repository contains tools and frameworks for code generation and evaluation using language models. The project leverages LangGraph and various language model APIs to provide a robust environment for code generation tasks.

## Features

- Code generation capabilities using language models
- Evaluation framework for generated code
- Integration with multiple language model APIs
- Configurable generation and evaluation pipelines
- Support for various programming languages and frameworks

## Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (recommended for optimal performance)
- Virtual environment (recommended)

## Installation

1. Create and activate a virtual environment:
```bash
virtualenv venv --python=3.10
source venv/bin/activate  
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
  "OPENAI_API_KEY": "your-api-key",
  "CODEEXEC_ENDPOINT": "http://localhost:5097/execute"
}
```

## Usage

The project provides two main functionalities:

### Code Generation

```bash
python3 run.py generator \
    --config_path=configs/jun_test_2.yaml \
    - run \
    - merge_json \
    - exit
```

### Code Evaluation

```bash
python run.py evaluator \
    --path=results/jun_test_2/results_merged_0.json \
    --gt_key=passed \
    - run \
    --k=[1] \
    --n=10
```

## Project Structure

```
autocode/
├── configs/          # Configuration files
├── src/             # Source code
├── templates/       # Template files
├── third_party/     # Third-party dependencies
├── api_keys.json    # API key configuration
├── run.py           # Main entry point
└── requirements.txt # Python dependencies
```

## Development

- Configuration files are stored in `configs/`
- Templates for code generation are in `templates/`

