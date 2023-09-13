## 0. Overview

- 학습을 위해서는 ```run.sh```의 ```--train_mode```를 활성화 시키면 됩니다. 활성화되지 않으면 학습이 완료된 모델을 기반으로 test가 이루어집니다.

- Task: NL -> PL, NL -> AST

  - NL->AST 가 기본 옵션이며, NL->PL을 하기 위해서는, 실행시 ```run.sh```의 ```--do_ast``` 옵션을 제거한 뒤 실행하시면 됩니다.
  
- Dataset: ```CodeSearchNet Corpus```

- Base tokenizer : [huggingface's microsoft/codebert-base tokenizer]
  
  - NL->AST 학습에서는 CodeSearchNet 데이터를 불러와 Treesitter를 활용하여 AST 파싱을 진행합니다.
   
  - 이때 AST node type은 자동으로 각 언어의 라이브러리에서 추출되어 사전학습 토크나이저의 vocab에 추가됩니다.
  
  - AST node type이 추가된 토크나이저는 ```src/pre_trained/fine_tune_tok```에 저장됩니다.
  
  - 이 때 해당 경로에 이미 저장된 토크나이저가 있다면, 불러온 뒤 학습에 활용합니다.

- CodeSearchNet의 6개 언어에 대해 Treesitter에서 제공하는 라이브러리를 빌드 후 사용하였습니다.

  - 빌드 결과는 src/build 폴더에 저장되어 있습니다.

## 1. Data Structure

- 학습을 위해서는 CodeSearchNet 디렉토리 내의 데이터(*.jsonl)은 사전에 준비가 필요합니다.
- src/data/cached 디렉토리 내 파일은 학습 과정에서 자동으로 생성되며, 해당 데이터는 입력으로 들어온 코드가 AST로 변환된 것입니다.

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

## 2.1 HW Environments

  - Ubuntu 20.04
  
  - GPU Nvidia 3080 이상

  - RAM 64GB 이상


## 2.2 Dependency 

- conda create -n autocode python=3.8.10
- conda activate autocode
- conda install numpy==1.19.2 pandas==1.1.5
- conda install openpyxl==3.0.7 xlrd==2.0.1 ipywidgets==7.6.3 jsonlines==2.0.0
- conda install pyyaml==5.4.1 
- pip install transformers==4.5.1 torch==1.7.1 scikit-learn==0.23.2 tree_sitter==0.20.1
- pip install tensorboardX
- [ERROR] packaging.version.InvalidVersion: Invalid version: '0.10.1,<0.11' 발생시 다음과 같이 해결

  - pip install packaging==21.3 (downgrading the packaging)

## 2.3 Setting for BLEU
 - human-eval이 설치되있어야 함
 - 참고: https://github.com/openai/human-eval
```sh
$ git clone https://github.com/openai/human-eval
$ pip install -e human-eval
```

## 2.4 CodeSearchNet Dataset 다운로드Setting for BLEU
 - CodeSearchNet 데이터셋은 공식 Github 홈페이지로부터 직접 다운로드
 - 참고: [https://github.com/openai/human-eval](https://github.com/github/CodeSearchNet)

   
## 3. How to Run

**Note**: ```[CLS]```, ```[SEP]``` 은 각각 ```<s>```, ```</s>```을 의미합니다.

**Note**: 학습 파라미터는 src/config.py에서 확인하실수 있습니다.

### Train
```sh
sh run.sh
```

### Inference
```sh
python src/inference.py --source "NL" --ckpt-path "CHECKPOINT_TO_LOAD"
```
