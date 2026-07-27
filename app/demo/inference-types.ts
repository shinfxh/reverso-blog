export type ModelId = "nano" | "small" | "base";
export type BrowserBackend = "webgpu" | "wasm";

export type ForecastResult = {
  model_id: ModelId;
  model: string;
  parameters: number;
  device: string;
  context_length: number;
  padded_context_length: number;
  horizon: number;
  forecast: number[];
};

export type ForecastWorkerRequest = {
  type: "forecast";
  requestId: number;
  modelId: ModelId;
  values: Float32Array;
  horizon: number;
  assetBase: string;
  forceWasm?: boolean;
};

export type ForecastWorkerResponse =
  | {
      type: "status";
      requestId: number;
      message: string;
    }
  | {
      type: "result";
      requestId: number;
      backend: BrowserBackend;
      elapsedMilliseconds: number;
      result: ForecastResult;
    }
  | {
      type: "error";
      requestId: number;
      message: string;
    };
