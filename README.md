# Reverso project page

Project page for **Reverso: Efficient Time Series Foundation Models for
Zero-shot Forecasting**.

- Paper: https://arxiv.org/abs/2602.17634
- Code: https://github.com/shinfxh/reverso
- Models: https://huggingface.co/shinfxh/reverso

## Local development

```bash
pnpm install
pnpm run dev
```

The demo runs Reverso-Nano and Reverso-Small directly in the browser. It uses
WebGPU when available and falls back to WebAssembly on the visitor's CPU.
No public inference backend or API URL is required.

The original FastAPI/PyTorch implementation remains available for local
reference and numerical comparisons:

```bash
python3 -m venv .venv
.venv/bin/pip install -r server/requirements.txt
pnpm run demo:api
```

Open `http://localhost:3000/demo/` after starting the frontend.

## Builds

```bash
pnpm run build
GITHUB_PAGES=true pnpm run build:github
```

The GitHub Actions workflow deploys the static export to GitHub Pages.

## Browser inference artifacts

The committed ONNX models in `public/models` are fixed-shape, float32 browser
graphs. Their checksums and PyTorch/ONNX comparison metrics are recorded in
`public/models/manifest.json`. Reverso Base is not shipped yet because its
initial browser graph did not meet the release size gate.

To regenerate and validate Nano and Small:

```bash
.venv/bin/pip install -r scripts/requirements-export.txt
pnpm run browser:models
```

The frontend build bundles the pinned ONNX Runtime Web files from
`node_modules` into the generated static assets.

The repository still contains the optional local FastAPI service and its
previous Render configuration, but the deployed GitHub Pages demo does not
contact either one.
