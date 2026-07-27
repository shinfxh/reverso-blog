#!/usr/bin/env python3
"""Export and validate Reverso models for ONNX Runtime Web.

The browser graph uses only static shapes and WebGPU-supported operators.
Every exported model is compared with both the original PyTorch model and
ONNX Runtime before its artifact and metadata are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import onnx
import onnxruntime as ort
import torch
from safetensors.torch import load_file
from torch import nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = REPOSITORY_ROOT / "server"
sys.path.insert(0, str(SERVER_DIR / "vendor"))

from reverso_torch.model import CNNBlock, Model, TorchDeltaNet  # noqa: E402


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    path: Path
    parameters: int


MODEL_SPECS = {
    "nano": ModelSpec("nano", SERVER_DIR / "model", 189_881),
    "small": ModelSpec(
        "small",
        SERVER_DIR / "models" / "small",
        550_161,
    ),
    "base": ModelSpec(
        "base",
        SERVER_DIR / "models" / "base",
        2_609_585,
    ),
}


class BrowserModel(nn.Module):
    def __init__(self, model: Model) -> None:
        super().__init__()
        self.model = model

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.model(context)


def load_model(spec: ModelSpec) -> tuple[Model, SimpleNamespace]:
    with (spec.path / "config.json").open() as handle:
        config = SimpleNamespace(**json.load(handle))

    model = Model(config)
    model.load_state_dict(
        load_file(spec.path / "model.safetensors"),
        strict=True,
    )
    model.eval()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != spec.parameters:
        raise RuntimeError(
            f"{spec.model_id} has {parameter_count:,} parameters; "
            f"expected {spec.parameters:,}"
        )
    return model, config


def make_browser_model(spec: ModelSpec) -> tuple[BrowserModel, SimpleNamespace]:
    model, config = load_model(spec)
    model.browser_export = True
    for module in model.modules():
        if isinstance(module, CNNBlock):
            module.forward = MethodType(CNNBlock.forward_browser, module)
        elif isinstance(module, TorchDeltaNet):
            module.browser_export = True
    browser_model = BrowserModel(model)
    browser_model.eval()
    return browser_model, config


def validation_inputs() -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(260217634)
    steps = torch.arange(2048, dtype=torch.float32)
    return [
        torch.randn(1, 2048, 1, generator=generator),
        (
            0.002 * steps
            + 1.7 * torch.sin(2 * torch.pi * steps / 48)
            + 0.4 * torch.sin(2 * torch.pi * steps / 9)
        ).reshape(1, 2048, 1),
        torch.linspace(-2, 3, 2048).reshape(1, 2048, 1),
    ]


def compare_browser_to_native(
    spec: ModelSpec,
    browser_model: BrowserModel,
) -> dict[str, float]:
    native_model, _ = load_model(spec)
    maximum_error = 0.0
    mean_error = 0.0

    with torch.inference_mode():
        for context in validation_inputs():
            expected = native_model(context)
            actual = browser_model(context)
            difference = (expected - actual).abs()
            maximum_error = max(maximum_error, difference.max().item())
            mean_error += difference.mean().item()

    return {
        "maximum_absolute_error_vs_pytorch": maximum_error,
        "mean_absolute_error_vs_pytorch": mean_error / 3,
    }


def export_onnx(
    browser_model: BrowserModel,
    output_path: Path,
    *,
    optimize: bool,
) -> None:
    sample = torch.zeros(1, 2048, 1, dtype=torch.float32)
    torch.onnx.export(
        browser_model,
        (sample,),
        output_path,
        input_names=["context"],
        output_names=["forecast"],
        opset_version=20,
        dynamo=True,
        external_data=False,
        optimize=optimize,
    )
    model = onnx.load(output_path)
    onnx.checker.check_model(model)


def compare_onnx_to_browser(
    browser_model: BrowserModel,
    output_path: Path,
) -> dict[str, float]:
    session = ort.InferenceSession(
        output_path,
        providers=["CPUExecutionProvider"],
    )
    maximum_error = 0.0
    mean_error = 0.0

    with torch.inference_mode():
        for context in validation_inputs():
            expected = browser_model(context).numpy()
            actual = session.run(
                ["forecast"],
                {"context": context.numpy()},
            )[0]
            difference = np.abs(expected - actual)
            maximum_error = max(maximum_error, float(difference.max()))
            mean_error += float(difference.mean())

    return {
        "maximum_absolute_error_onnx_vs_export_model": maximum_error,
        "mean_absolute_error_onnx_vs_export_model": mean_error / 3,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "models",
        nargs="*",
        choices=MODEL_SPECS,
        default=["nano"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "public" / "models",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {"format": 1, "models": {}}
    )

    for model_id in args.models:
        spec = MODEL_SPECS[model_id]
        print(f"Preparing {model_id}...", flush=True)
        browser_model, config = make_browser_model(spec)
        pytorch_metrics = compare_browser_to_native(spec, browser_model)

        output_path = args.output_dir / f"reverso-{model_id}.onnx"
        export_onnx(
            browser_model,
            output_path,
            optimize=model_id != "base",
        )
        onnx_metrics = compare_onnx_to_browser(browser_model, output_path)

        manifest["models"][model_id] = {
            "file": output_path.name,
            "bytes": output_path.stat().st_size,
            "sha256": sha256(output_path),
            "parameters": spec.parameters,
            "context_length": config.seq_len,
            "output_length": config.output_token_len,
            **pytorch_metrics,
            **onnx_metrics,
        }
        print(json.dumps(manifest["models"][model_id], indent=2))

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
