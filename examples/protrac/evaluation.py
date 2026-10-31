import platform
from pathlib import Path

import timm
import torch
from omegaconf import OmegaConf

from cofai.engine.outputs import write_eval_outputs


def save_results(cfg, results):
    output = Path(cfg.output)
    output.mkdir(parents=True, exist_ok=True)
    run_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    run_cfg.environment = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "timm": timm.__version__,
    }
    write_eval_outputs(
        str(output),
        name=f"protrac-{cfg.task}",
        description=f"ProTRAC online calibration for {cfg.task}",
        results=results,
        cfg=run_cfg,
    )


def infer(model, images, task, real_codec=False, **kwargs):
    if real_codec:
        coded = model.compress(images, tasks=[task], **kwargs)
        return model.decompress(coded, tasks=[task], **kwargs)[task]
    _, output = model.forward_test(images, tasks=[task], **kwargs)
    return output[task]
