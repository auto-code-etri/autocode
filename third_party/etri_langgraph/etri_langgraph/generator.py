import asyncio
import json
import logging
import os
from pathlib import Path
from traceback import format_exc
from typing import Any, List, Optional

import yaml
from etri_langgraph.config import Config
from etri_langgraph.graph import Graph
from etri_langgraph.loader import Loader
from langchain_core.documents import Document
from pydantic import BaseModel
from tqdm.asyncio import tqdm_asyncio

import wandb

"""registry """
from etri_langgraph.utils import registry  # isort:skip
from etri_langgraph.model import *
from etri_langgraph.prompt import *


class Generator(BaseModel):
    verbose: bool = False
    do_save: bool = True
    api_keys_path: str = "api_keys.json"
    target_dataset_name: str = "target"
    example_dataset_name: str = "example"
    wandb_on: bool = False
    langfuse_on: bool = False
    rerun: bool = False
    max_concurrency: int = 4

    run_name: str = None  # if None, config_path.stem is used
    config_path: Path = None
    config: Config = None

    # private variables
    output_dir: Path = None
    results_dir: Path = None
    datasets: dict = {}
    target_dataset: dict = {}
    example_dataset: dict = {}
    graph: Graph = None

    def __init__(self, **data):
        super().__init__(**data)

        if self.verbose:
            logging.basicConfig(level=logging.INFO)

        self._load_config()
        self._init_result_dir()
        self._load_api_keys()
        self._load_datasets()
        self._init_wandb()
        self._compile_graph()

    def _load_config(self):
        if self.config_path is not None:
            self.config = Config(path=self.config_path)
        elif self.config is None:
            raise ValueError("Either config_path or config should be provided")
        else:
            pass

    def _init_result_dir(self):
        if self.run_name is None:
            self.run_name = self.config_path.stem
        if self.do_save:
            self.output_dir = Path(f"results/{self.run_name}")
            self.results_dir = self.output_dir / "results"
            self.results_dir.mkdir(parents=True, exist_ok=True)

    def _load_api_keys(self):
        api_keys = json.loads(Path(self.api_keys_path).read_text())
        for k, v in api_keys.items():
            os.environ[k] = v

    def _load_datasets(self):
        loader = Loader(config=self.config)
        self.datasets = loader.run().result
        self.target_dataset = self.datasets.get(self.target_dataset_name, {})
        self.example_dataset = self.datasets.get(self.example_dataset_name, {})
        del self.datasets[self.target_dataset_name]
        if self.example_dataset_name in self.datasets:
            del self.datasets[self.example_dataset_name]

    def _init_wandb(self):
        wandb.require("core")

        mode = "disabled"
        if self.wandb_on:
            logging.info("Wandb mode is online")
            mode = "online"

        wandb.init(
            mode=mode,
            entity=os.environ.get("WANDB_ENTITY", None),
            project=os.environ.get("WANDB_PROJECT", None),
            name=self.run_name,
            notes=self.config.description,
        )

        wandb.config.update(self.config.model_dump())

    def _compile_graph(self):
        self.graph = Graph(
            config=self.config.graph,
            examples=self.example_dataset,
            etc_datasets=self.datasets,
        ).run()

    def run(
        self,
        n: Optional[int] = None,
        ids: Optional[list] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ):
        targets = self.target_dataset
        if n is not None:
            targets = {k: v for k, v in list(targets.items())[:n]}
        elif start is not None and end is not None:
            targets = {k: v for k, v in list(targets.items())[start:end]}
        elif ids is not None:
            targets = {k: v for k, v in targets.items() if k in ids}
        else:
            pass

        asyncio.run(self._run(targets))

        return self

    async def _run(
        self,
        targets: dict,
    ):
        tasks = []
        sem = asyncio.Semaphore(self.max_concurrency)
        for id, target in targets.items():
            task = self._run_one(id, target, sem)
            tasks.append(task)

        await tqdm_asyncio.gather(*tasks)

    async def _run_one(
        self,
        id: str,
        target: dict,
        sem: asyncio.Semaphore,
    ):
        """
        Run the target and save the result as json file
        """
        done = False
        id = str(id)
        if self.do_save:
            path = self.results_dir / f"{str(id).replace('/', '_')}.json"
            if path.exists() and not self.rerun:
                logging.info(f"{id} already exists. Skipping...")
                try:
                    result = json.loads(path.read_text())
                    if "error" in result[0]:
                        raise Exception("Error in previous run")
                    done = True
                except Exception as e:
                    logging.error(f"Error in loading {id}")
                    logging.error(format_exc())

        if not done:
            async with sem:
                if self.langfuse_on:
                    from langfuse.callback import CallbackHandler

                    langfuse_handler = CallbackHandler()
                    config = {"callbacks": [langfuse_handler]}
                else:
                    config = {}

                config.update({"id": id, "verbose": self.verbose})
                try:
                    result = await self.graph.ainvoke([target])
                    logging.info(f"Done: {id}")
                except Exception as e:
                    logging.error(f"Error in running {id}")
                    result = [{"error": format_exc()}]

        if self.do_save:
            self._save_json(id.replace("/", "_"), result)
            self._save_yaml(id.replace("/", "_"), result)
            self._save_files(id.replace("/", "_"), result)

        return result

    def _save_json(self, id: str, result: dict):
        """
        Save result as json file
        """
        path = self.results_dir / f"{id}.json"
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.results_dir / f"{id}.json", "w") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

    def _save_yaml(self, id: str, result: dict):
        """
        Save result as yaml file
        """
        path = self.results_dir / f"{id}.yaml"
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.results_dir / f"{id}.yaml", "w") as f:
            yaml.dump(
                result, f, default_style="|", default_flow_style=False, sort_keys=False
            )

    def _save_files(self, id: str, result: str):
        output_dir = self.results_dir / id
        def _rec_save_files(result, dir, key):
            dir.mkdir(parents=True, exist_ok=True)
            key = str(key)

            if isinstance(result, str):
                file = dir / f"{key}.txt"
                with open(file, "w") as f:
                    f.write(result)
                wandb.save(file, base_path=self.results_dir, policy="now")
            elif isinstance(result, list):
                for i in range(len(result)):
                    _rec_save_files(result[i], dir / key, i)
            elif isinstance(result, dict):
                for k, v in result.items():
                    _rec_save_files(v, dir / key, k)
            elif isinstance(result, Document):
                file = dir / f"{key}.txt"
                with open(file, "w") as f:
                    f.write(result.to_json())
                wandb.save(file, base_path=self.results_dir, policy="now")
            else:
                try:
                    file = dir / f"{key}.json"
                    with open(file, "w") as f:
                        json.dump(result, f, indent=4, ensure_ascii=False)
                    wandb.save(file, base_path=self.results_dir, policy="now")
                except Exception as e:
                    logging.error(f"Error in saving files: {e}")

        _rec_save_files(result, output_dir, "result")

    def merge_json(self):
        """
        Merge json files distributed by problem into one file
        """
        data = {}
        for file in os.listdir(f"{self.output_dir}/results"):
            if file.endswith(".json"):
                with open(f"{self.output_dir}/results/{file}", "r") as f:
                    id = file.split(".")[0]
                    data[id] = json.load(f)

        # sort by id and turn into list
        data = dict(sorted(data.items(), key=lambda x: x[0]))
        data = list(data.values())

        # max length of the result
        max_len = max([len(d) for d in data])

        dump_data = [{} for _ in range(len(data))]
        for i in range(max_len):
            for j, d in enumerate(data):
                if isinstance(d, list):
                    if i < len(d):
                        dump_data[j].update(d[i])
                    else:
                        dump_data[j].update({"max_depth": len(d)})
                elif isinstance(d, dict):
                    dump_data[j] = d
                else:
                    raise ValueError("Invalid data type")

            filename = f"{self.output_dir}/results_merged_{i}.json"
            with open(filename, "w") as f:
                json.dump(dump_data, f, indent=4, ensure_ascii=False)

        return self

    def exit(self):
        """
        Exit the generator
        """
        pass

    async def astream_user_input(
        self,
        nl_query: str,
        event_names: Optional[List[str]] = None,
    ):
        """
        Run user input
        """
        target = {"prompt": nl_query}

        gen = self.graph.astream_events(
            [target],
            version="v2",
            include_names=event_names,
        )
        async for result in gen:
            if result["event"] == "on_chain_end":
                yield result["data"]["output"]
