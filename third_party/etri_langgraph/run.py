import fire

from etri_langgraph.config import Config
from etri_langgraph.generator import Generator
from etri_langgraph.loader import Loader

if __name__ == "__main__":
    fire.Fire(
        {
            "config": Config,
            "loader": Loader,
            "generator": Generator,
        }
    )
