import re
from typing import Dict, List, Tuple, Union
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import BaseTransformOutputParser

# implement this that can wrap code block parser into a runnable langgraph
class CodeBlockParser(BaseTransformOutputParser[str]):
    def _code_block_parser(self, input_data: Union[str, List, Dict]):
        """
        Parses code blocks from markdown text.
        Adapts logic from expand_langchain's code_block_runner.
        """
        def func_str(text: str):
            # Match ```lang ... ``` blocks
            pattern = r"```[a-z]*\n(.*?)```"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                matches = re.findall(pattern, text, re.DOTALL)
                if matches:
                    return matches[-1]
                else:
                    return match.group(1)
            else:
                # Handle unclosed blocks
                pattern = r"```[a-z]*\n(.*?)$"
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    matches = re.findall(pattern, text, re.DOTALL)
                    if matches:
                        return matches[-1]
                    else:
                        return match.group(1)
                else:
                    return text

        if isinstance(input_data, str):
            return func_str(input_data)
        elif isinstance(input_data, List):
            return [func_str(text) for text in input_data]
        elif isinstance(input_data, Dict):
            return {key: func_str(text) for key, text in input_data.items()}
        else:
            raise ValueError(f"input type {type(input_data)} is not supported")


    def parse(self, text: str) -> str:
        return self._code_block_parser(text)


class FindParser(BaseTransformOutputParser[Union[List[str], Dict[str, List[str]]]]):
    patterns: List[str]

    def parse(self, text: Union[str, Dict[str, str]]) -> Union[List[str], Dict[str, List[str]]]:
        def func_str(input_text: str) -> List[str]:
            """
            find all occurrences of patterns
            the order of result is the order of occurrence in the original text
            """
            result = []
            for pattern in self.patterns:
                for match in re.finditer(pattern, input_text):
                    cursor, string = match.start(), match.group()
                    result.append((cursor, string))

            result = [string for cursor, string in sorted(result)]
            return result

        if isinstance(text, str):
            return func_str(text)
        elif isinstance(text, dict):
             return {key: func_str(value) for key, value in text.items()}
        else:
             raise ValueError(f"Unsupported input type for FindParser: {type(text)}")

class DistributeParser(BaseTransformOutputParser[Dict[str, str]]):
    landmarks: List[Tuple[str, str]]

    def parse(self, text: str) -> Dict[str, str]:
        _landmarks = {key: landmark for key, landmark in self.landmarks}

        # find landmarks
        cursors = {}
        for key, landmark in _landmarks.items():
            match = re.search(landmark, text)
            start = match.start() if match else len(text)
            end = match.end() if match else len(text)
            cursors[key] = (start, end)

        # extract strings
        result = {}
        for key, (start, end) in cursors.items():
            # find nearest other start
            next_starts = [v[0] for k, v in cursors.items() if k != key]
            next_starts.append(len(text))
            next_starts = [v for v in next_starts if v > start]
            _end = min(next_starts, default=len(text))

            # extract string
            result[key] = text[end:_end].rstrip()

        if result == {}:
            result = {"default": text}

        return result