from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import torch
import torch.utils.mobile_optimizer
from pydantic import BaseModel
from seisbench.models import PhaseNet

logging.basicConfig(level=logging.INFO)


class DefaultArgs(BaseModel):
    P_threshold: float = 0.3
    S_threshold: float = 0.3
    blinding: Tuple[int, int]


class ModelMetaData(BaseModel):
    version: int = 1
    pre_trained: str
    docstring: str
    default_args: DefaultArgs


class PhaseNetExporter:
    input = torch.zeros(1, 3, 3001)

    def __init__(self, pre_trained: str, version_str: str = "latest"):
        self.model = PhaseNet.from_pretrained(pre_trained, version_str=version_str)
        self.model.eval()
        self.pre_trained = pre_trained

    def export_model(self, path: Path, optimize_for_mobil: bool = False) -> None:
        # We have to export the trace before we can create the script.
        traced_script = torch.jit.trace(self.model, self.input)
        script = torch.jit.script(traced_script)

        if optimize_for_mobil:
            script = torch.utils.mobile_optimizer.optimize_for_mobile(script)

        if not self.model._weights_metadata:
            raise ValueError("Model Metadata is undefined (model._weights_metadata)")

        weights_metadata: dict = self.model._weights_metadata
        version = weights_metadata.get("version", 1)
        weights_metadata["pre_trained"] = self.pre_trained

        meta_data = ModelMetaData(**weights_metadata)

        filename = path / f"PhaseNet-{self.pre_trained}-v{version}.pt"
        logging.info("Saving TorchScript model to %s", filename)

        torch.jit.save(
            script,
            str(filename),
            _extra_files={"weights_metadata.json": meta_data.json(indent=2)},
        )


def main() -> None:
    from argparse import ArgumentParser

    parser = ArgumentParser(
        prog="seisbench-PhaseNet",
        description="Export pre-trained seisbench PhaseNet models slim TorchScript.",
    )
    parser.add_argument("out", type=Path, help="Directory for the TorchScript models.")
    ns = parser.parse_args()

    out_dir: Path = ns.out

    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    for pre_trained in PhaseNet.list_pretrained():
        if pre_trained == "obs":
            continue
        exporter = PhaseNetExporter(pre_trained)
        exporter.export_model(out_dir)
