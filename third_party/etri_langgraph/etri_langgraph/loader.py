import json
import logging
import os
import pprint
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset, load_dataset
from elasticsearch import Elasticsearch, helpers
from pydantic import BaseModel
from tqdm import tqdm

from etri_langgraph.config import Config, DatasetConfig, SourceConfig

logger = logging.getLogger(__name__)


class Loader(BaseModel):
    config: Config = None
    config_path: str = None
    api_keys_path: str = "api_keys.json"

    result: Any = None

    def __init__(self, **data):
        super().__init__(**data)

        self._load_config()
        self._load_api_keys()

    def _load_config(self):
        if self.config_path is not None:
            self.config = Config2(path=self.config_path)
        elif self.config is None:
            raise ValueError("Either config_path or config should be provided")
        else:
            pass

    def _load_api_keys(self):
        api_keys = json.loads(Path(self.api_keys_path).read_text())
        for k, v in api_keys.items():
            os.environ[k] = v

    def run(self):
        sources = self.load_sources()     #huggingface에서 source data 다운, question_id로 sort
        self.result = self._load_datasets(sources)          #source를 dataset 형태로 변경    self.result가 중요
#        print(self.result)
        return self

    def load_sources(self):
        sources = {}
        for source in self.config.source:
            source: SourceConfig
            name = source.name

            if source.type == "huggingface":    #소스만 가져옴
                path = source.kwargs.get("path")
                sort_key = source.kwargs.get("sort_key")
                split = source.kwargs.get("split")
                load_dataset_kwargs = source.kwargs.get("load_dataset_kwargs", {})
                dataset = load_dataset(path, **load_dataset_kwargs)[split]
                sources[name] = dataset.sort(sort_key)

            elif source.type == "json":
                path = source.kwargs.get("path")
                sort_key = source.kwargs.get("sort_key")
                data_json = json.loads(Path(path).read_text())
                sources[name] = Dataset.from_list(data_json).sort(sort_key)

            elif source.type == "jsonl":
                path = source.kwargs.get("path")
                sort_key = source.kwargs.get("sort_key")
                data_jsonl = [
                    json.loads(line)
                    for line in Path(path).read_text().split("\n")
                    if line
                ]
                sources[name] = Dataset.from_list(data_jsonl).sort(sort_key)

            elif source.type == "yaml":
                path = source.kwargs.get("path")
                sort_key = source.kwargs.get("sort_key")
                data_yaml = yaml.load(Path(path).read_text(), Loader=yaml.FullLoader)
                sources[name] = Dataset.from_list(data_yaml).sort(sort_key)
                
            elif source.type == "user_input":
                sources[name] = None
                
            else:
                raise ValueError(f"Unsupported source type: {source.type}")

        return sources

    def _load_datasets(self, sources):              #
        datasets = {}
        for dataset in self.config.dataset:   #dataset 정보
            dataset: DatasetConfig

            name = dataset.name
            if dataset.type == "dict":
                result = _load_dict(sources, **dataset.kwargs)

            elif dataset.type == "user_input":
                result = None

            else:
                raise ValueError(f"Unknown dataset type: {dataset.type}")

            if not dataset.remove:     #target name
                datasets[name] = result

        return datasets

    def save(self, path):
        if not Path(path).parent.exists():
            Path(path).parent.mkdir(parents=True)

        with open(path, "w") as f:
            json.dump(self.result, f, indent=2, ensure_ascii=False)

        return self

    def exit(self):
        pass

def _load_dict(
    sources,
    primary_key,
    fields,
    query: str = None,
    cache_dir: str = None,
    custom_lambda: str = None,
):
    config = {
        "primary_key": primary_key,
        "fields": fields,
        "query": query,
    }
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        config_path = cache_dir / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                cache_config = json.load(f)
            if cache_config == config:
                data_path = cache_dir / "data.json"
                if data_path.exists():
                    with open(data_path, "r") as f:
                        return json.load(f)

    result = {}

    primary_field = list(filter(lambda x: x.get("name") == primary_key, fields))[0]
    ids = sources[primary_field.get("source")][primary_field.get("key")]
    for i, id in tqdm(enumerate(ids)):
        result[id] = {}
        for field in fields:
            source = sources[field.get("source")]
            result[id][field.get("name")] = source[i][field.get("key")]

    if custom_lambda is not None:
        try:
            func_obj = eval(custom_lambda)
        except:
            local_namespace = {}
            exec(custom_lambda, globals(), local_namespace)
            func_obj = local_namespace["func"]

        result = {k: func_obj(v) for k, v in result.items()}

    if query is not None:
        from tinydb import TinyDB, where
        from tinydb.storages import MemoryStorage

        db = TinyDB(storage=MemoryStorage)
        db.insert_multiple(list(result.values()))
        result = db.search(eval(query, {"where": where}))
        result = {r[primary_key]: r for r in result}

    if cache_dir is not None:
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True)

        with open(cache_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        with open(cache_dir / "data.json", "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    return result
