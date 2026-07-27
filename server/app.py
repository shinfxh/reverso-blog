from __future__ import annotations

import json
import math
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from safetensors.torch import load_file

SERVER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SERVER_DIR / "vendor"))

from reverso_torch.forecast import forecast_flip_equivariant  # noqa: E402
from reverso_torch.model import Model  # noqa: E402

ModelId = Literal["nano", "small", "base"]

MODEL_SPECS = {
    "nano": {
        "name": "Reverso-Nano",
        "path": SERVER_DIR / "model",
        "parameters": 189_881,
    },
    "small": {
        "name": "Reverso-Small",
        "path": SERVER_DIR / "models" / "small",
        "parameters": 550_161,
    },
    "base": {
        "name": "Reverso",
        "path": SERVER_DIR / "models" / "base",
        "parameters": 2_609_585,
    },
}


class ForecastRequest(BaseModel):
    values: list[float] = Field(min_length=32, max_length=2048)
    horizon: int = Field(default=48, ge=1, le=480)
    model: ModelId = "nano"


class ReversoRuntime:
    def __init__(self, model_id: ModelId) -> None:
        spec = MODEL_SPECS[model_id]
        config_path = spec["path"] / "config.json"
        weights_path = spec["path"] / "model.safetensors"

        with config_path.open() as handle:
            self.config = SimpleNamespace(**json.load(handle))

        self.model_id = model_id
        self.name = spec["name"]
        requested_device = os.environ.get("REVERSO_DEVICE", "cpu")
        if requested_device == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model = Model(self.config)
        self.model.load_state_dict(load_file(weights_path), strict=True)
        self.model.to(self.device).eval()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        if self.parameter_count != spec["parameters"]:
            raise RuntimeError(
                f"{self.name} has {self.parameter_count:,} parameters; "
                f"expected {spec['parameters']:,}."
            )
        self.lock = threading.Lock()

        self._run([0.0] * 2048, 1)

    def _synchronize(self) -> None:
        if self.device.type == "mps":
            torch.mps.synchronize()

    def _run(self, values: list[float], horizon: int) -> list[float]:
        context = torch.tensor(
            values, dtype=torch.float32, device=self.device
        ).reshape(1, -1, 1)

        if context.shape[1] < self.config.seq_len:
            pad_length = self.config.seq_len - context.shape[1]
            context = torch.cat(
                [context[:, :1, :].expand(-1, pad_length, -1), context], dim=1
            )

        with torch.inference_mode():
            output = forecast_flip_equivariant(
                self.model,
                context,
                prediction_length=horizon,
                seq_len=self.config.seq_len,
                output_token_len=self.config.output_token_len,
                use_amp=False,
            )
        self._synchronize()
        return output[0, :, 0].float().cpu().tolist()

    def predict(self, values: list[float], horizon: int) -> list[float]:
        if not all(math.isfinite(value) for value in values):
            raise ValueError("All values must be finite numbers.")

        with self.lock:
            return self._run(values, horizon)


class RuntimeRegistry:
    def __init__(self) -> None:
        self.runtimes: dict[ModelId, ReversoRuntime] = {}
        self.lock = threading.Lock()

    def get(self, model_id: ModelId) -> ReversoRuntime:
        with self.lock:
            if model_id not in self.runtimes:
                self.runtimes[model_id] = ReversoRuntime(model_id)
            return self.runtimes[model_id]


runtimes = RuntimeRegistry()

app = FastAPI(title="Reverso local inference", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://shinfxh.github.io",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ready",
        "models": [
            {
                "id": model_id,
                "name": spec["name"],
                "parameters": spec["parameters"],
                "loaded": model_id in runtimes.runtimes,
            }
            for model_id, spec in MODEL_SPECS.items()
        ],
        "flip_equivariance": True,
    }


@app.post("/api/forecast")
def create_forecast(request: ForecastRequest) -> dict[str, object]:
    runtime = runtimes.get(request.model)
    try:
        prediction = runtime.predict(request.values, request.horizon)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return {
        "model_id": runtime.model_id,
        "model": runtime.name,
        "parameters": runtime.parameter_count,
        "device": runtime.device.type,
        "context_length": len(request.values),
        "padded_context_length": runtime.config.seq_len,
        "horizon": request.horizon,
        "flip_equivariance": True,
        "forecast": prediction,
    }
