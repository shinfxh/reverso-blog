"use client";

import {
  ChangeEvent,
  PointerEvent as ReactPointerEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import giftEvalData from "./gift-eval-examples.json";
import {
  ForecastResult,
  ForecastWorkerRequest,
  ForecastWorkerResponse,
  ModelId,
} from "./inference-types";

type HistoryView = 192 | 512 | 2048;

type DrawingPoint = {
  x: number;
  y: number;
};

type GiftEvalExample = {
  id: string;
  label: string;
  values: number[];
  futureValues: number[];
};

const MODEL_OPTIONS: {
  id: ModelId;
  label: string;
  parameters: number;
  browserReady: boolean;
}[] = [
  {
    id: "nano",
    label: "Reverso-Nano",
    parameters: 189881,
    browserReady: true,
  },
  {
    id: "small",
    label: "Reverso-Small",
    parameters: 550161,
    browserReady: true,
  },
  {
    id: "base",
    label: "Reverso",
    parameters: 2609585,
    browserReady: false,
  },
];

const DRAWING_WIDTH = 960;
const DRAWING_HEIGHT = 260;
const DRAWING_SEQUENCE_LENGTH = 2048;
const MIN_DRAWING_SPAN = 24;

const GIFT_EVAL_EXAMPLES = giftEvalData.examples as GiftEvalExample[];
const DEFAULT_EXAMPLE = GIFT_EVAL_EXAMPLES[0];
const SYNTHETIC_EXAMPLES = [
  { id: "sine", label: "Sine wave" },
  { id: "sawtooth", label: "Sawtooth" },
];

const makePreset = (name: string, length = 2048) => {
  const values = Array.from({ length }, (_, index) => {
    if (name === "sawtooth") {
      return ((index % 40) / 40) * 8 - 4;
    }
    return 4 * Math.sin((2 * Math.PI * index) / 40);
  });
  return values.map((value) => value.toFixed(5)).join(", ");
};

const parseValues = (raw: string) =>
  raw
    .split(/[\s,;]+/)
    .map((value) => value.trim())
    .filter(Boolean)
    .map(Number);

const pointerToDrawingPoint = (
  event: ReactPointerEvent<SVGSVGElement>,
): DrawingPoint => {
  const bounds = event.currentTarget.getBoundingClientRect();
  const x = ((event.clientX - bounds.left) / bounds.width) * DRAWING_WIDTH;
  const y = ((event.clientY - bounds.top) / bounds.height) * DRAWING_HEIGHT;
  return {
    x: Math.max(0, Math.min(DRAWING_WIDTH, x)),
    y: Math.max(0, Math.min(DRAWING_HEIGHT, y)),
  };
};

const resampleDrawing = (
  points: DrawingPoint[],
  length = DRAWING_SEQUENCE_LENGTH,
) => {
  if (
    points.length < 2 ||
    points[points.length - 1].x - points[0].x < MIN_DRAWING_SPAN
  ) {
    return [];
  }

  const startX = points[0].x;
  const endX = points[points.length - 1].x;
  let segmentIndex = 0;

  return Array.from({ length }, (_, index) => {
    const targetX = startX + (index / (length - 1)) * (endX - startX);
    while (
      segmentIndex < points.length - 2 &&
      points[segmentIndex + 1].x < targetX
    ) {
      segmentIndex += 1;
    }

    const left = points[segmentIndex];
    const right = points[segmentIndex + 1];
    const span = Math.max(right.x - left.x, 1e-6);
    const mix = Math.max(0, Math.min(1, (targetX - left.x) / span));
    const interpolatedY = left.y + (right.y - left.y) * mix;

    return 0.5 - interpolatedY / DRAWING_HEIGHT;
  });
};

function ForecastChart({
  context,
  forecast,
  modelName,
  historyPoints,
  truth,
}: {
  context: number[];
  forecast: number[];
  modelName: string;
  historyPoints: HistoryView;
  truth?: number[];
}) {
  const width = 900;
  const height = 360;
  const padding = 36;
  const visibleContext = context.slice(-historyPoints);
  const visibleTruth = truth?.slice(0, forecast.length) ?? [];
  const allValues = [...visibleContext, ...forecast, ...visibleTruth];
  const chartLength =
    visibleContext.length + Math.max(forecast.length, visibleTruth.length);
  const min = Math.min(...allValues);
  const max = Math.max(...allValues);
  const range = Math.max(max - min, 1e-6);
  const x = (index: number) =>
    padding + (index / Math.max(chartLength - 1, 1)) * (width - 2 * padding);
  const y = (value: number) =>
    height - padding - ((value - min) / range) * (height - 2 * padding);
  const path = (values: number[], offset: number) =>
    values
      .map(
        (value, index) =>
          `${index === 0 ? "M" : "L"} ${x(index + offset).toFixed(2)} ${y(
            value,
          ).toFixed(2)}`,
      )
      .join(" ");
  const boundary = x(visibleContext.length - 1);

  return (
    <figure className="demo-chart">
      <svg
        role="img"
        aria-label={`Historical time-series values followed by the ${modelName} forecast${visibleTruth.length > 0 ? " and held-out true values" : ""}`}
        viewBox={`0 0 ${width} ${height}`}
      >
        <line
          className="chart-axis"
          x1={padding}
          x2={width - padding}
          y1={height - padding}
          y2={height - padding}
        />
        <line
          className="chart-boundary"
          x1={boundary}
          x2={boundary}
          y1={padding}
          y2={height - padding}
        />
        <path className="chart-context" d={path(visibleContext, 0)} />
        <path
          className="chart-forecast"
          d={path(
            [visibleContext[visibleContext.length - 1], ...forecast],
            visibleContext.length - 1,
          )}
        />
        {visibleTruth.length > 0 && (
          <path
            className="chart-truth"
            d={path(
              [visibleContext[visibleContext.length - 1], ...visibleTruth],
              visibleContext.length - 1,
            )}
          />
        )}
        <text x={padding} y={22}>
          history
        </text>
        <text x={Math.min(boundary + 12, width - 110)} y={22}>
          forecast
        </text>
      </svg>
      <div className="chart-legend" aria-hidden="true">
        <span>
          <span className="chart-legend-line is-history" />
          History
        </span>
        <span>
          <span className="chart-legend-line is-forecast" />
          {modelName}
        </span>
        {visibleTruth.length > 0 && (
          <span>
            <span className="chart-legend-line is-truth" />
            True value
          </span>
        )}
      </div>
      <figcaption>
        The final {visibleContext.length} input points are shown in black;
        {` ${modelName}’s`} zero-shot forecast is shown in blue
        {visibleTruth.length > 0
          ? ", and the held-out Gift-Eval continuation is shown in red."
          : "."}
      </figcaption>
    </figure>
  );
}

export default function DemoPage() {
  const [exampleId, setExampleId] = useState(DEFAULT_EXAMPLE.id);
  const [rawValues, setRawValues] = useState(
    DEFAULT_EXAMPLE.values.join(", "),
  );
  const [modelId, setModelId] = useState<ModelId>("small");
  const [horizon, setHorizon] = useState(48);
  const [historyPoints, setHistoryPoints] = useState<HistoryView>(192);
  const [result, setResult] = useState<ForecastResult | null>(null);
  const [status, setStatus] = useState("Ready");
  const [error, setError] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const [drawingPoints, setDrawingPoints] = useState<DrawingPoint[]>([]);
  const [isDrawing, setIsDrawing] = useState(false);
  const workerRef = useRef<Worker | null>(null);
  const requestIdRef = useRef(0);
  const values = useMemo(() => parseValues(rawValues), [rawValues]);
  const drawingValues = useMemo(
    () => resampleDrawing(drawingPoints),
    [drawingPoints],
  );
  const drawingPath = useMemo(
    () =>
      drawingPoints
        .map(
          (point, index) =>
            `${index === 0 ? "M" : "L"} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`,
        )
        .join(" "),
    [drawingPoints],
  );
  const selectedModel =
    MODEL_OPTIONS.find((model) => model.id === modelId) ?? MODEL_OPTIONS[0];
  const selectedGiftExample = GIFT_EVAL_EXAMPLES.find(
    (example) => example.id === exampleId,
  );

  useEffect(
    () => () => {
      workerRef.current?.terminate();
      workerRef.current = null;
    },
    [],
  );

  const resetOutput = () => {
    setResult(null);
    setStatus("Ready");
    setError("");
  };

  const chooseExample = (event: ChangeEvent<HTMLSelectElement>) => {
    const nextExampleId = event.target.value;
    const giftEvalExample = GIFT_EVAL_EXAMPLES.find(
      (example) => example.id === nextExampleId,
    );
    setExampleId(nextExampleId);
    setRawValues(
      giftEvalExample
        ? giftEvalExample.values.join(", ")
        : makePreset(nextExampleId),
    );
    setHistoryPoints(192);
    resetOutput();
  };

  const uploadCsv = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setExampleId("custom");
    setRawValues(await file.text());
    resetOutput();
  };

  const startDrawing = (event: ReactPointerEvent<SVGSVGElement>) => {
    if (isRunning || (event.pointerType === "mouse" && event.button !== 0)) {
      return;
    }
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    setDrawingPoints([pointerToDrawingPoint(event)]);
    setIsDrawing(true);
  };

  const continueDrawing = (event: ReactPointerEvent<SVGSVGElement>) => {
    if (!isDrawing) return;
    event.preventDefault();
    const nextPoint = pointerToDrawingPoint(event);
    setDrawingPoints((currentPoints) => {
      const lastPoint = currentPoints[currentPoints.length - 1];
      if (!lastPoint || nextPoint.x - lastPoint.x < 1.5) {
        return currentPoints;
      }
      return [...currentPoints, nextPoint];
    });
  };

  const finishDrawing = (event: ReactPointerEvent<SVGSVGElement>) => {
    if (!isDrawing) return;
    setIsDrawing(false);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  const useDrawing = () => {
    if (drawingValues.length === 0) return;
    setExampleId("custom");
    setRawValues(
      drawingValues.map((value) => value.toFixed(6)).join(", "),
    );
    setHistoryPoints(2048);
    setResult(null);
    setError("");
    setStatus(
      `Drawing converted to ${DRAWING_SEQUENCE_LENGTH} input points`,
    );
  };

  const runForecast = async () => {
    setError("");
    setResult(null);
    if (values.length < 32 || values.length > 2048) {
      setError("Provide between 32 and 2,048 numeric values.");
      return;
    }
    if (values.some((value) => !Number.isFinite(value))) {
      setError("The input contains a value that is not a number.");
      return;
    }

    if (!selectedModel.browserReady) {
      setStatus("Unavailable");
      setError(
        "Reverso Base is still being optimized for safe browser delivery.",
      );
      return;
    }

    setStatus(`Preparing ${selectedModel.label} locally…`);
    setIsRunning(true);
    try {
      const worker =
        workerRef.current ??
        new Worker(new URL("./inference.worker.ts", import.meta.url), {
          type: "module",
        });
      workerRef.current = worker;
      requestIdRef.current += 1;
      const requestId = requestIdRef.current;
      const inputValues = new Float32Array(values);
      const assetBase = new URL("../", window.location.href).href;

      const response = await new Promise<
        Extract<ForecastWorkerResponse, { type: "result" }>
      >((resolve, reject) => {
        const handleMessage = (
          event: MessageEvent<ForecastWorkerResponse>,
        ) => {
          const message = event.data;
          if (!message || message.requestId !== requestId) return;
          if (message.type === "status") {
            setStatus(message.message);
            return;
          }
          worker.removeEventListener("message", handleMessage);
          worker.removeEventListener("error", handleWorkerError);
          if (message.type === "error") {
            reject(new Error(message.message));
          } else {
            resolve(message);
          }
        };
        const handleWorkerError = (event: ErrorEvent) => {
          worker.removeEventListener("message", handleMessage);
          worker.removeEventListener("error", handleWorkerError);
          reject(
            new Error(
              event.message || "The browser inference worker stopped.",
            ),
          );
        };

        worker.addEventListener("message", handleMessage);
        worker.addEventListener("error", handleWorkerError);
        const request: ForecastWorkerRequest = {
          type: "forecast",
          requestId,
          modelId,
          values: inputValues,
          horizon,
          assetBase,
          forceWasm:
            new URLSearchParams(window.location.search).get("runtime") ===
            "wasm",
        };
        worker.postMessage(request, [inputValues.buffer]);
      });

      setResult(response.result);
      setStatus(
        `Forecast complete · ${response.result.device} · ${(
          response.elapsedMilliseconds / 1000
        ).toFixed(2)}s`,
      );
    } catch (caught) {
      setStatus("Could not run");
      setError(
        caught instanceof Error
          ? caught.message
          : "The inference service is unavailable.",
      );
    } finally {
      setIsRunning(false);
    }
  };

  const downloadForecast = () => {
    if (!result) return;
    const rows = ["step,forecast", ...result.forecast.map((value, index) => `${index + 1},${value}`)];
    const blob = new Blob([rows.join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${result.model_id}-forecast.csv`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <main className="demo-main">
      <header className="demo-header">
        <p className="label">
          <a href="../">← Reverso paper</a>
        </p>
        <h1>
          <strong>Reverso</strong>
          <span> live forecast demo</span>
        </h1>
        <p className="subtitle">
          Compare the released Reverso models on a univariate time series.
          Nothing is fitted to your data.
        </p>
      </header>

      <section className="demo-intro" aria-labelledby="how-it-works">
        <h2 id="how-it-works">Zero-shot inference</h2>
        <p>
          Draw a sequence, choose an example, or paste 32–2,048 comma-separated
          observations. Shorter histories are padded to the 2,048-point context,
          and each model predicts up to 480 future points.
        </p>
      </section>

      <section className="drawing-studio" aria-labelledby="drawing-title">
        <div className="drawing-heading">
          <div>
            <p className="label">Sketch an input</p>
            <h2 id="drawing-title">Draw your own sequence</h2>
          </div>
        </div>

        <svg
          className={`drawing-canvas${isDrawing ? " is-drawing" : ""}${isRunning ? " is-disabled" : ""}`}
          viewBox={`0 0 ${DRAWING_WIDTH} ${DRAWING_HEIGHT}`}
          role="img"
          aria-label="Interactive time-series drawing area"
          onPointerDown={startDrawing}
          onPointerMove={continueDrawing}
          onPointerUp={finishDrawing}
          onPointerCancel={finishDrawing}
          onLostPointerCapture={() => setIsDrawing(false)}
        >
          <g className="drawing-grid" aria-hidden="true">
            {[0.25, 0.5, 0.75].map((fraction) => (
              <line
                key={`vertical-${fraction}`}
                x1={DRAWING_WIDTH * fraction}
                x2={DRAWING_WIDTH * fraction}
                y1={0}
                y2={DRAWING_HEIGHT}
              />
            ))}
            {[0.25, 0.5, 0.75].map((fraction) => (
              <line
                key={`horizontal-${fraction}`}
                x1={0}
                x2={DRAWING_WIDTH}
                y1={DRAWING_HEIGHT * fraction}
                y2={DRAWING_HEIGHT * fraction}
              />
            ))}
          </g>
          <text className="drawing-scale" x={10} y={18}>
            +0.5
          </text>
          <text className="drawing-scale" x={10} y={DRAWING_HEIGHT / 2 - 8}>
            0
          </text>
          <text className="drawing-scale" x={10} y={DRAWING_HEIGHT - 10}>
            −0.5
          </text>
          <text
            className="drawing-time-label"
            x={DRAWING_WIDTH - 70}
            y={DRAWING_HEIGHT - 10}
          >
            time →
          </text>
          {drawingPoints.length === 0 && (
            <text
              className="drawing-placeholder"
              x={DRAWING_WIDTH / 2}
              y={DRAWING_HEIGHT / 2}
              textAnchor="middle"
            >
              Press and drag to draw
            </text>
          )}
          {drawingPath && <path className="drawing-stroke" d={drawingPath} />}
        </svg>

        <div className="drawing-footer">
          <p aria-live="polite">
            {drawingValues.length > 0
              ? `${DRAWING_SEQUENCE_LENGTH}-point sequence ready`
              : drawingPoints.length > 0
                ? "Keep drawing farther to the right"
                : "Mouse, pen, and touch supported"}
          </p>
          <div className="drawing-actions">
            <button
              type="button"
              className="drawing-clear"
              disabled={drawingPoints.length === 0 || isRunning}
              onClick={() => {
                setDrawingPoints([]);
                setIsDrawing(false);
              }}
            >
              Clear
            </button>
            <button
              type="button"
              className="drawing-use"
              disabled={drawingValues.length === 0 || isRunning}
              onClick={useDrawing}
            >
              Use drawing as input
            </button>
          </div>
        </div>
      </section>

      <section className="demo-workbench" aria-label="Forecast workbench">
        <div className="demo-controls">
          <label>
            Model
            <select
              value={modelId}
              disabled={isRunning}
              onChange={(event) => {
                setModelId(event.target.value as ModelId);
                resetOutput();
              }}
            >
              {MODEL_OPTIONS.map((model) => (
                <option
                  key={model.id}
                  value={model.id}
                  disabled={!model.browserReady}
                >
                  {model.label} · {model.parameters.toLocaleString()} parameters
                  {!model.browserReady ? " · optimizing for browser" : ""}
                </option>
              ))}
            </select>
          </label>

          <label>
            Example series
            <select
              value={exampleId}
              disabled={isRunning}
              onChange={chooseExample}
            >
              {exampleId === "custom" && (
                <option value="custom">Custom input</option>
              )}
              <optgroup label="Gift-Eval benchmark">
                {GIFT_EVAL_EXAMPLES.map((example) => (
                  <option key={example.id} value={example.id}>
                    {example.label}
                  </option>
                ))}
              </optgroup>
              <optgroup label="Synthetic">
                {SYNTHETIC_EXAMPLES.map((example) => (
                  <option key={example.id} value={example.id}>
                    {example.label}
                  </option>
                ))}
              </optgroup>
            </select>
          </label>

          <label>
            Observations
            <textarea
              value={rawValues}
              disabled={isRunning}
              onChange={(event) => {
                setExampleId("custom");
                setRawValues(event.target.value);
                resetOutput();
              }}
              spellCheck={false}
              aria-describedby="input-count"
            />
          </label>
          <p id="input-count" className="input-count">
            {values.length.toLocaleString()} numeric points
          </p>

          <div className="demo-control-row">
            <label>
              Forecast horizon
              <select
                value={horizon}
                disabled={isRunning}
                onChange={(event) => {
                  setHorizon(Number(event.target.value));
                  resetOutput();
                }}
              >
                <option value={12}>12 points</option>
                <option value={24}>24 points</option>
                <option value={48}>48 points</option>
                <option value={96}>96 points</option>
                <option value={192}>192 points</option>
                <option value={336}>336 points</option>
                <option value={480}>480 points</option>
              </select>
            </label>
            <label>
              History shown
              <select
                value={historyPoints}
                disabled={isRunning}
                onChange={(event) =>
                  setHistoryPoints(
                    Number(event.target.value) as HistoryView,
                  )
                }
              >
                <option value={192}>Last 192 points</option>
                <option value={512}>Last 512 points</option>
                <option value={2048}>Last 2,048 points</option>
              </select>
            </label>
          </div>

          <label className="file-label">
            Upload CSV
            <input
              type="file"
              accept=".csv,.txt,text/csv,text/plain"
              disabled={isRunning}
              onChange={uploadCsv}
            />
          </label>

          <button
            type="button"
            disabled={isRunning || !selectedModel.browserReady}
            onClick={runForecast}
          >
            {isRunning
              ? `Running ${selectedModel.label}…`
              : `Run ${selectedModel.label} locally`}
          </button>
          <p className="run-status" aria-live="polite">{status}</p>
          <p className="runtime-note">
            Uses WebGPU when available and falls back to your CPU. Your values
            never leave this browser.
          </p>
          {error && <p className="demo-error">{error}</p>}
        </div>

        <div className="demo-output" aria-live="polite">
          {result ? (
            <>
              <div className="result-summary">
                <p><strong>{result.model}</strong><span>{result.parameters.toLocaleString()} parameters</span></p>
                <p><strong>{result.horizon}</strong><span>forecast points</span></p>
                <p><strong>{Math.min(historyPoints, values.length).toLocaleString()}</strong><span>history shown</span></p>
              </div>
              <ForecastChart
                context={values}
                forecast={result.forecast}
                modelName={result.model}
                historyPoints={historyPoints}
                truth={selectedGiftExample?.futureValues}
              />
              <div className="forecast-values">
                <h2>Output values</h2>
                <ol>
                  {result.forecast.slice(0, 12).map((value, index) => (
                    <li key={index}>
                      <span>t+{index + 1}</span>
                      <code>{value.toFixed(6)}</code>
                    </li>
                  ))}
                </ol>
                {result.forecast.length > 12 && (
                  <p>Showing 12 of {result.forecast.length} predicted points.</p>
                )}
                <button type="button" className="secondary-button" onClick={downloadForecast}>
                  Download forecast CSV
                </button>
              </div>
            </>
          ) : (
            <div className="empty-output">
              <p>Forecast output will appear here.</p>
              <span>
                The first request downloads the selected model and starts the
                private on-device runtime.
              </span>
            </div>
          )}
        </div>
      </section>

      <footer>
        <p>
          This demo runs the released Reverso weights locally in your browser.
          WebGPU is used when available, with a WebAssembly CPU fallback.
        </p>
        <p>
          <a href="https://github.com/unit8co/darts/pull/3061">Darts PR #3061</a>
          {" · "}
          <a href="https://huggingface.co/shinfxh/models">Model weights</a>
        </p>
      </footer>
    </main>
  );
}
