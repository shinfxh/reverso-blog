/// <reference lib="webworker" />

import * as ort from "onnxruntime-web/webgpu";
import {
  BrowserBackend,
  ForecastResult,
  ForecastWorkerRequest,
  ForecastWorkerResponse,
  ModelId,
} from "./inference-types";

type LoadedSession = {
  backend: BrowserBackend;
  session: ort.InferenceSession;
};

const workerScope = self as unknown as DedicatedWorkerGlobalScope;
const sessions = new Map<string, LoadedSession>();
const modelMetadata = {
  nano: {
    file: "reverso-nano.onnx",
    label: "Reverso-Nano",
    parameters: 189_881,
  },
  small: {
    file: "reverso-small.onnx",
    label: "Reverso-Small",
    parameters: 550_161,
  },
  base: {
    file: "reverso-base.onnx",
    label: "Reverso",
    parameters: 2_609_585,
  },
} satisfies Record<
  ModelId,
  { file: string; label: string; parameters: number }
>;

const CONTEXT_LENGTH = 2048;
const OUTPUT_LENGTH = 48;

ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;

const post = (message: ForecastWorkerResponse) => {
  workerScope.postMessage(message);
};

const postStatus = (
  requestId: number,
  message: string,
) => {
  post({ type: "status", requestId, message });
};

const createSession = async (
  modelId: ModelId,
  assetBase: string,
  requestId: number,
  forceWasm: boolean,
): Promise<LoadedSession> => {
  const sessionKey = `${modelId}:${forceWasm ? "wasm" : "auto"}`;
  const existing = sessions.get(sessionKey);
  if (existing) return existing;

  const metadata = modelMetadata[modelId];
  const modelUrl = new URL(`models/${metadata.file}`, assetBase).href;
  const webGpuAvailable = !forceWasm && "gpu" in navigator;

  if (webGpuAvailable) {
    postStatus(requestId, `Loading ${metadata.label} with WebGPU…`);
    try {
      const session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ["webgpu"],
        graphOptimizationLevel: "all",
      });
      const loaded = { backend: "webgpu", session } as const;
      sessions.set(sessionKey, loaded);
      return loaded;
    } catch (error) {
      console.warn("WebGPU initialization failed; using CPU.", error);
    }
  }

  postStatus(
    requestId,
    webGpuAvailable
      ? "WebGPU could not load this model; switching to CPU…"
      : "WebGPU is unavailable; loading the CPU runtime…",
  );
  const session = await ort.InferenceSession.create(modelUrl, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  const loaded = { backend: "wasm", session } as const;
  sessions.set(sessionKey, loaded);
  return loaded;
};

const runChunk = async (
  session: ort.InferenceSession,
  context: Float32Array,
): Promise<Float32Array> => {
  const input = new ort.Tensor(
    "float32",
    context,
    [1, CONTEXT_LENGTH, 1],
  );
  try {
    const outputs = await session.run({ context: input });
    const output = outputs.forecast;
    if (!output || !(output.data instanceof Float32Array)) {
      throw new Error("The browser model returned an invalid forecast.");
    }
    const values = new Float32Array(output.data);
    output.dispose();
    return values;
  } finally {
    input.dispose();
  }
};

const padContext = (values: Float32Array): Float32Array => {
  const context = new Float32Array(CONTEXT_LENGTH);
  const padding = CONTEXT_LENGTH - values.length;
  context.fill(values[0], 0, padding);
  context.set(values, padding);
  return context;
};

const appendChunk = (
  context: Float32Array,
  chunk: Float32Array,
): Float32Array => {
  const next = new Float32Array(CONTEXT_LENGTH);
  next.set(context.subarray(OUTPUT_LENGTH), 0);
  next.set(chunk, CONTEXT_LENGTH - OUTPUT_LENGTH);
  return next;
};

const validateRequest = (request: ForecastWorkerRequest) => {
  if (
    !Object.hasOwn(modelMetadata, request.modelId) ||
    request.modelId === "base"
  ) {
    throw new Error("That browser model is not available.");
  }
  if (!(request.values instanceof Float32Array)) {
    throw new Error("The browser model requires float32 input.");
  }
  const assetBase = new URL(request.assetBase);
  if (assetBase.origin !== workerScope.location.origin) {
    throw new Error("Model assets must load from this site.");
  }
  if (request.values.length < 32 || request.values.length > CONTEXT_LENGTH) {
    throw new Error("Provide between 32 and 2,048 numeric values.");
  }
  if (
    !Number.isInteger(request.horizon) ||
    request.horizon < 1 ||
    request.horizon > 480
  ) {
    throw new Error("Forecast horizon must be between 1 and 480.");
  }
  for (const value of request.values) {
    if (!Number.isFinite(value)) {
      throw new Error("The input contains a value that is not a number.");
    }
  }
};

const forecast = async (
  request: ForecastWorkerRequest,
): Promise<ForecastWorkerResponse> => {
  validateRequest(request);
  const started = performance.now();
  const loaded = await createSession(
    request.modelId,
    request.assetBase,
    request.requestId,
    request.forceWasm === true,
  );
  const metadata = modelMetadata[request.modelId];
  const steps = Math.ceil(request.horizon / OUTPUT_LENGTH);
  const predictions: number[] = [];
  let context = padContext(request.values);

  for (let step = 0; step < steps; step += 1) {
    postStatus(
      request.requestId,
      `Running ${metadata.label} locally · chunk ${step + 1} of ${steps}…`,
    );
    const positive = await runChunk(loaded.session, context);
    const negativeContext = Float32Array.from(
      context,
      (value) => -value,
    );
    const negative = await runChunk(loaded.session, negativeContext);
    const chunk = Float32Array.from(
      positive,
      (value, index) => (value - negative[index]) / 2,
    );
    predictions.push(...chunk);
    context = appendChunk(context, chunk);
  }

  const result: ForecastResult = {
    model_id: request.modelId,
    model: metadata.label,
    parameters: metadata.parameters,
    device:
      loaded.backend === "webgpu"
        ? "WebGPU"
        : "WebAssembly CPU",
    context_length: request.values.length,
    padded_context_length: CONTEXT_LENGTH,
    horizon: request.horizon,
    forecast: predictions.slice(0, request.horizon),
  };

  return {
    type: "result",
    requestId: request.requestId,
    backend: loaded.backend,
    elapsedMilliseconds: performance.now() - started,
    result,
  };
};

workerScope.addEventListener("message", async (event) => {
  const request = event.data as ForecastWorkerRequest;
  if (!request || request.type !== "forecast") return;

  try {
    post(await forecast(request));
  } catch (error) {
    console.error(error);
    post({
      type: "error",
      requestId: request.requestId,
      message:
        error instanceof Error
          ? error.message
          : "Browser inference failed.",
    });
  }
});
