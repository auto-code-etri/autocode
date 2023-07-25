## 0. Overview

- Task: NL -> PL, NL -> AST

    NL->AST 가 기본 옵션이며, NL->PL을 하기 위해서는 config에서 "do_ast" 옵션을 false로 변경해야 합니다.
  
- Dataset: CodeSearchNet Python corpus

    |  Train  | Valid  |  Test  |
    | :-----: | :----: | :----: |
    | 360,000 | 24,000 | 20,000 |

- base tokenizer : [huggingface's microsoft/codebert-base tokenizer]
  
   NL->AST 학습에서는 CodeSearchNet 데이터를 불러와 Treesitter를 활용하여 AST 파싱을 진행합니다.
   
   이때 파싱한 AST의 노드 종류(node type)을 동적으로 사전학습 토크나이저의 vocab에 추가하고 학습에 활용합니다.
  
   AST 노드 종류가 추가된 토크나이저는 ```pre_trained/fine_tune_tok```에 저장됩니다.
  
   이 때 해당 경로에 이미 저장된 토크나이저가 있다면, 불러온 뒤 학습에 활용합니다.

- CodeSearchNet의 6개 언어에 대해 Treesitter에서 제공하는 라이브러리를 빌드 후 사용하였습니다.

  Build 폴더에 Treesitter에서 제공하는 라이브러리를 clone하고 build 후 사용하시면 됩니다. 

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
  └─ CodeSearchNet/ 
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

**Note**: ```[CLS]```, ```[SEP]``` 은 각각 ```<s>```, ```</s>```을 의미합니다.

**Note**: CodeSearchNet Corpus의 Code data는 dataset.py/cache_processed_data()를 통해 AST로 변환되며, 학습에 활용됩니다.

**Note**: 학습 파라미터는 config.py에서 확인하실수 있습니다.

### Train
```sh
python3 main.py 
```

### Inference
```sh
python inference.py --source "NL" --ckpt-path "CHECKPOINT_TO_LOAD"
```
