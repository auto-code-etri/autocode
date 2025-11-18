[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![ETRI License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/auto-code-etri/autocode">
    <img src="image/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">PULSE</h3>

  <p align="center">
    Pipeline for Unified LLM Software Engineering
    <br />
    <a href="https://github.com/auto-code-etri/autocode/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/auto-code-etri/autocode/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributors</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#Acknowledgment">Acknowledgment</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

**PULSE** is a large language model–based framework that automatically generates **expert-level, high-quality source code** from natural language requirements.

This project provides the following key features:

- Analyzes user requirements to generate **structured, testable, and maintainable code**, producing results that go beyond simple code snippets to practical, production-ready outputs.
- Supports a **structured pipeline and multiple language model environments**, making it suitable for both research and real-world development.

<div align="center">
  <a href="https://github.com/auto-code-etri/autocode">
    <img src="image/overview_autocode.png" alt="Overview">
  </a>
</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

- [![Lang][Langchain]][Langchain-url]
- [![Dock][Docker]][Docker-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (recommended for optimal performance)
- Virtual environment (recommended)

## Installation

1. Install uv (A fast Python package installer and resolver):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create and activate a virtual environment:

```bash
python -m venv test
source test/bin/activate
```

3. Install dependencies:

```bash
uv pip install -r requirements.txt
git submodule update --init --recursive
uv pip install -e third_party/etri_langgraph
```

## Configuration

1. export OPENAI_API_KEY in your .bashrc file in HOME directory

```bash
vi ~/.bashrc
### add the following line for setting OPENAI_API key ###
export OPENAI_API_KEY='sk-(your key)'
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->

## Usage

This project provides a code generation environment based on the PULSE / CodeNet workflow.

### Input data format

The default input file is `inputs/data.json`. An example is:

```json
{
  "id": "HumanEval0_Problem",
  "entry_point": "has_close_elements",
  "prompt": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
  "gold_tc": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n"
}
```

| Field         | Description                                                                                |
| ------------- | ------------------------------------------------------------------------------------------ |
| `id`          | Identifier of the input example                                                            |
| `entry_point` | Name of the target function that the model is expected to implement                        |
| `prompt`      | Problem statement and function signature provided to the model as input                    |
| `gold_tc`     | Reference test code (test harness) used to check the correctness of the generated solution |

You can create your own input file following the same structure and pass it via the --input_file option.

Basic usage

Run the pipeline with the default settings:

```bash
python3 run.py
```

This is equivalent to:

```bash
python3 run.py \
  --input_file inputs/data.json \
  --output_dir outputs
```

Specify input file and output directory

Use --input_file to specify another input JSON file and --output_dir to change the location of the result files:

```bash
python3 run.py \
  --input_file inputs/custom_data.json \
  --output_dir results/run_001
```

After execution, the generated result will be stored in:

```bash
<output_dir>/output.json
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->

## Contributors

<a href="https://github.com/auto-code-etri/autocode/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=auto-code-etri/autocode" alt="contrib.rocks image" />
</a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->

## License

Copyright _On-Device AI Model Research Laboratory, ETRI_.

All rights reserved. For more details, see `LICENSE`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->

## Acknowledgment

> This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.2022-0-00995, Automated reliable source code generation from natural language descriptions)

> 이 논문은 2025년도 정부(과학기술정보통신부)의 재원으로 정보통신기획평가원의 지원을 받아 수행된 연구임 (No.2022-0-00995, 자연어로 기술된 요구사항에서 전문 개발자 수준의 고품질 코드를 자동 생성하는 기술 개발)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/auto-code-etri/autocode.svg?style=for-the-badge
[contributors-url]: https://github.com/auto-code-etri/autocode/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/auto-code-etri/autocode.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/auto-code-etri/autocode.svg?style=for-the-badge
[stars-url]: https://github.com/auto-code-etri/autocode/stargazers
[issues-shield]: https://img.shields.io/github/issues/auto-code-etri/autocode.svg?style=for-the-badge
[issues-url]: https://github.com/auto-code-etri/autocode/issues
[license-shield]: https://img.shields.io/github/license/auto-code-etri/autocode.svg?style=for-the-badge
[license-url]: https://github.com/auto-code-etri/autocode/blob/master/LICENSE
[Langchain]: https://img.shields.io/badge/LangChain-ffffff?logo=langchain&logoColor=green
[Langchain-url]: https://www.langchain.com/
[Docker]: https://img.shields.io/badge/docker-257bd6?style=for-the-badge&logo=docker&logoColor=white
[Docker-url]: https://www.docker.com/
