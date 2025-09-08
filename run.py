import fire
from etri_langgraph.generator import Generator

if __name__ == "__main__":
    fire.Fire(
        {
            "generator": Generator,
        }
    )