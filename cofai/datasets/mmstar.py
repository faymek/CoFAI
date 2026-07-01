import io
import os
import base64
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from PIL import Image


class MMStarDataset:
    """
    Loads MMStar Benchmark dataset from a TSV file.
    """

    def __init__(
        self, data_path: str = "MMStar.tsv", prompt_suffix: Optional[str] = ""
    ):
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                "Data TSV not found. Please download from: https://opencompass.openxlab.space/utils/VLMEval/MMStar.tsv and place it in the current directory."
            )

        self.data = pd.read_csv(data_path, sep="\t")
        self.data["index"] = self.data["index"].astype(str)
        self.option_cols = ["A", "B", "C", "D"]
        self.prompt_suffix = prompt_suffix

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def _decode_image(image_str: str) -> Image.Image:
        """
        Decodes a base64-encoded image string into a PIL Image object.
        """
        if image_str.startswith("data:image"):
            image_str = image_str.split(",", 1)[1]
        image_bytes = base64.b64decode(image_str)
        return Image.open(io.BytesIO(image_bytes))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a dictionary containing the following keys:
            {
                "img": np.ndarray,          # RGB image, HWC uint8
                "vqa": dict,                # VQA input and annotations
                "meta": dict,               # image-only metadata
            }
        """
        item = self.data.iloc[idx]
        options = {c: item[c] for c in self.option_cols if pd.notna(item[c])}
        prompt_parts = []

        # Delete "Hint:" line (no use) from the question text
        question_lines = item["question"].splitlines()
        filtered_question = ""
        for line in question_lines:
            if line.strip().startswith("Hint:"):
                continue
            filtered_question += line + "\n"
        filtered_question = filtered_question.strip()
        prompt_parts.append(f"Question: {filtered_question}")

        if options:
            opt_str = "\n".join([f"{k}. {v}" for k, v in options.items()])
            prompt_parts.append("Options:\n" + opt_str + "\n")
            prompt_parts.append(self.prompt_suffix)
        prompt = "\n".join(prompt_parts)

        # Decode image to PIL
        image = self._decode_image(str(item["image"]))

        def opt_str(col):
            return str(item[col]) if col in item and pd.notna(item[col]) else None

        result = {
            "img": np.asarray(image.convert("RGB"), dtype=np.uint8),
            "vqa": {
                "prompt": prompt,
                "question": str(item["question"]),
                "options": options,
                "answer": str(item["answer"]),
                "category": opt_str("category"),
                "l2_category": opt_str("l2_category"),
                "bench": opt_str("bench"),
            },
            "meta": {
                "ori_size": image.size[::-1],
                "img_name": str(item["index"]),
            },
        }
        return result
