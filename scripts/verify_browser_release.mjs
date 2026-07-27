#!/usr/bin/env node

import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readdir, readFile, stat } from "node:fs/promises";
import { dirname, extname, join } from "node:path";
import { fileURLToPath } from "node:url";

const repositoryRoot = dirname(fileURLToPath(import.meta.url));
const projectRoot = dirname(repositoryRoot);
const publicModelsDirectory = join(projectRoot, "public", "models");
const outputDirectory = join(projectRoot, "out");
const expectedModelIds = ["nano", "small"];

const sha256 = async (path) =>
  createHash("sha256").update(await readFile(path)).digest("hex");

const walk = async (directory) => {
  const entries = await readdir(directory, { withFileTypes: true });
  const paths = [];
  for (const entry of entries) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) {
      paths.push(...(await walk(path)));
    } else {
      paths.push(path);
    }
  }
  return paths;
};

const manifest = JSON.parse(
  await readFile(join(publicModelsDirectory, "manifest.json"), "utf8"),
);
assert.equal(manifest.format, 1);
assert.deepEqual(Object.keys(manifest.models).sort(), expectedModelIds);

for (const modelId of expectedModelIds) {
  const metadata = manifest.models[modelId];
  const sourcePath = join(publicModelsDirectory, metadata.file);
  const outputPath = join(outputDirectory, "models", metadata.file);
  const sourceStats = await stat(sourcePath);
  const outputStats = await stat(outputPath);

  assert.equal(sourceStats.size, metadata.bytes, `${modelId} source size`);
  assert.equal(outputStats.size, metadata.bytes, `${modelId} output size`);
  assert.equal(await sha256(sourcePath), metadata.sha256, `${modelId} source`);
  assert.equal(await sha256(outputPath), metadata.sha256, `${modelId} output`);
}

const outputPaths = await walk(outputDirectory);
assert.equal(
  outputPaths.filter(
    (path) =>
      extname(path) === ".wasm" &&
      path.includes("ort-wasm-simd-threaded"),
  ).length,
  1,
  "The release must contain one ONNX Runtime WebAssembly binary",
);
assert.equal(
  outputPaths.some((path) => path.endsWith("reverso-base.onnx")),
  false,
  "The oversized Base model must not be deployed",
);

const publicTextExtensions = new Set([
  ".css",
  ".html",
  ".js",
  ".json",
  ".mjs",
  ".txt",
  ".xml",
]);
const sensitiveReleasePattern =
  /127\.0\.0\.1|localhost:8000|NEXT_PUBLIC_REVERSO_API_URL|\/api\/forecast|onrender\.com|hf\.space|BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}/i;

for (const path of outputPaths) {
  if (!publicTextExtensions.has(extname(path))) continue;
  const contents = await readFile(path, "utf8");
  assert.doesNotMatch(contents, sensitiveReleasePattern, path);
}

console.log(
  `Verified ${expectedModelIds.length} browser models, one runtime, and ${outputPaths.length} public files.`,
);
