from pathlib import Path

from dotenv import load_dotenv
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]


def load_config(path):
    load_dotenv(ROOT / ".env", override=True, encoding="utf-8")
    return OmegaConf.load(path)
