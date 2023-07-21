## 0. Overview

- Task: NL -> PL, NL -> AST

    NS->AST 가 기본 옵션이며, NL->PL을 하기 위해서는 config에서 "do_ast"를 변경한다.
  
- Dataset: CodeSearchNet Python corpus

    |  Train  | Valid  |  Test  |
    | :-----: | :----: | :----: |
    | 360,000 | 24,000 | 20,000 |

- base tokenizer : [huggingface's microsoft/codebert-base tokenizer]

## 1. Data Structure
- 학습을 위해서는 CodeSearchNet 디렉토리 내의 데이터(*.jsonl)은 준비가 필요합니다.
- data/cached 디렉토리 내 파일은 학습 과정에서 자동으로 생성되며, 해당 데이터는 입력으로 들어온 코드가 AST로 변환된 것입니다.
```sh
src/
  └─ data/
    └─ cached/ # tokenized indice with special tokens
        ├─ cached_train.jsonl
        ├─ cached_valid.jsonl
        └─ cached_test.jsonl
  └─ CodeSearchNet/ # Train, valid, and test datasets need to be prepared for each of the six languages in CodeSearchNet.
    └─ Language(python, java, ruby ...)/
        ├─ train.jsonl
        ├─ valid.jsonl
        └─ test.jsonl
```

## 2. Environment

- numpy==1.19.2
- pandas==1.1.5
- transformers==4.5.1
- torch==1.7.1
- openpyxl==3.0.7
- xlrd==2.0.1
- ipywidgets==7.6.3
- sklearn==0.23.2
- jsonlines==2.0.0
- pyyaml==5.4.1
- tree_sitter==0.20.1

## 3. How to Run

**Note**: In this implementation, ```[CLS]``` and ```[SEP]``` tokens mean ```<s>``` and ```</s>``` respectively.

**Note**: The codes in CodeSearchNet are converted into AST and utilized for training through the dataset.py/cache_processed_data() function.

**Note**: Configuration of this project -> config.py

### Train

python3 main.py 

### Inference

python inference.py --source "NL" --ckpt-path "CHECKPOINT_TO_LOAD"
