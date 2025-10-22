[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/auto-code-etri/autocode">
    <img src="logo.png" alt="Logo" width="80" height="80">
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
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

[![Project Overview][Overview]]

**PULSE** is a large language model–based framework that automatically generates **expert-level, high-quality source code** from natural language requirements.  

This project provides the following key features:  
- Analyzes user requirements to generate **structured, testable, and maintainable code**, producing results that go beyond simple code snippets to practical, production-ready outputs.  
- Supports a **structured pipeline and multiple language model environments**, making it suitable for both research and real-world development.  

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Built With

* [![Lang][Langchain]][Langchain-url]
* [![Dock][Docker]][Docker-url]


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (recommended for optimal performance)
- Virtual environment (recommended)

### Installation

1. Create and activate a virtual environment:
```bash
python -m venv test
source test/bin/activate 
```

2. Install dependencies:
```bash
pip install -r requirements.txt
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

The project provides two main functionalities:

### Code Generation

#### GPT-4
```bash
python3 run.py
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTRIBUTING -->
## Contributors

<a href="https://github.com/auto-code-etri/autocode/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=auto-code-etri/autocode" alt="contrib.rocks image" />
</a>

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
[Langchain]: https://img.shields.io/badge/LangChain-ffffff?logo=langchain&logoColor=green
[Langchain-url]: https://www.langchain.com/
[Docker]: https://img.shields.io/badge/docker-257bd6?style=for-the-badge&logo=docker&logoColor=white
[Docker-url]: https://www.docker.com/
[Overview]: overview.autocode.png
