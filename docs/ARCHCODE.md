# ArchCode
This is a PULSE adaptation of the official ArchCode implementation ([paper](https://aclanthology.org/2024.acl-long.730/), [code](https://github.com/ldilab/ArchCode): `networks/archcode`
Following the example langgraph network, `ArchCodeNet` is added on `networks/archcode/archcode.py`.
The detailed configuration of the graph is described at `networks/archcode/ArchCode_etri.yaml`.
The prompt templates for the executions are added at:
```bash
# basic prompt templates
/workspace/autocode/templates/prompt/archcode
# oneshot example
/workspace/autocode/templates/example 
# evaluation prompt
/workspace/autocode/templates/eval/archcode
```

## ⚠️ Before Running
### Execution Environment
Please prepare the execution environment `third_party/CodeExecContainer`.
As you run `git submodule update --init --recursive`, you can get the designated repositories.
If it does not appear, please run again.
After cloning the repository, run the following command:
```bash
sh run.sh
```
Now you can run the generated code via `http://localhost:5097`

### environment variables
Please add `.env` at the root of this project.
```bash
OPENAI_API_KEY=xxxx
CODEEXEC_ENDPOINT=http://localhost:5097/execute
```

## Run
```python
from networks.archcode.archcode import ArchCodeNet

net = ArchCodeNet()
net.compile()
result = net.run(state={'input_data': "implement sorting algorithm", "code_n": 10})
```

You can run the example execution at:
```bash
python tests/test_archcode.py
```

This script will run for `implement sorting algorithm` by generating 10 candidate codes and ranking them by functional requirements and non-functional requirements.