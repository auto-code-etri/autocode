import os
from itertools import zip_longest
from typing import Any, Dict, List, Optional

from etri_langgraph.utils.registry import node_registry, BaseNode
from langchain_community.utilities.requests import JsonRequestsWrapper

@node_registry(name="execute")
class ExecuteNode(BaseNode):
    def __init__(
        self,
        key: str,
        code_key: str,
        testcase_key: Optional[str] = None,
        stdin_key: Optional[str] = None,
        **kwargs,
    ):
        self.key = key
        self.code_key = code_key
        self.testcase_key = testcase_key
        self.stdin_key = stdin_key
        self.kwargs = kwargs

    async def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        result = {self.key: []}
        requests_wrapper = JsonRequestsWrapper()

        for target, testcase in zip_longest(
            data[self.code_key],
            data.get(self.testcase_key, []),
            fillvalue={},
        ):
            if isinstance(target, str):
                response = requests_wrapper.post(
                    os.environ["CODEEXEC_ENDPOINT"],
                    data={
                        "code": target,
                        "stdin": testcase.get(self.stdin_key, ""),
                        **self.kwargs,
                    },
                )
                result[self.key].append(response["output"])

            elif isinstance(target, list):
                outputs = []
                for _target, _testcase in zip_longest(
                    target,
                    testcase,
                    fillvalue={},
                ):
                    response = requests_wrapper.post(
                        os.environ["CODEEXEC_ENDPOINT"],
                        data={
                            "code": _target,
                            "stdin": _testcase.get(self.stdin_key, ""),
                            **self.kwargs,
                        },
                    )
                    outputs.append(response["output"])

                result[self.key].append(outputs)
            else:
                raise ValueError("Invalid input type")

        data.update(result)
        return data
