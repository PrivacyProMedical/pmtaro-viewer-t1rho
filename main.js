import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { loadPyodide } from 'pyodide';
import {
  createPyodideModuleRuntime,
} from '../../../utils/apiLoader.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const runtime = createPyodideModuleRuntime({
  moduleDir: __dirname,
  loadPyodide,
  sourceLabel: '@pmt/t1rho api/main.py',
  cExtensionPackages: [
    'setuptools',
    'numpy',
  ],
  wheelPackages: [
    'pydicom',
  ],
});

async function init(ctx) {
  return runtime.init(ctx);
}

async function fitT1rhoMap(instance1, tsl1_ms, instance2, tsl2_ms, instance3, tsl3_ms, instance4, tsl4_ms, outputDir) {
  await init();

  const resolvedArgs = runtime.validateCall('fit_t1rho_map', [
    instance1,
    tsl1_ms,
    instance2,
    tsl2_ms,
    instance3,
    tsl3_ms,
    instance4,
    tsl4_ms,
    outputDir,
  ]);

  const hostOutputDir = resolvedArgs[8] || join(__dirname, 'api', 't1rho_outputs');
  fs.mkdirSync(hostOutputDir, { recursive: true });

  const stagedArgs = [
    runtime.stageFilePayload(resolvedArgs[0], 'instance_1'),
    resolvedArgs[1],
    runtime.stageFilePayload(resolvedArgs[2], 'instance_2'),
    resolvedArgs[3],
    runtime.stageFilePayload(resolvedArgs[4], 'instance_3'),
    resolvedArgs[5],
    runtime.stageFilePayload(resolvedArgs[6], 'instance_4'),
    resolvedArgs[7],
    hostOutputDir,
  ];

  const result = await runtime.invokePythonFunction('fit_t1rho_map', stagedArgs);
  runtime.validateResult('fit_t1rho_map', result);

  const outputPayload = runtime.bridgeOutputFilePayloadToHost(result.output_dcm, hostOutputDir);
  const previewPayload = runtime.bridgeOutputFilePayloadToHost(result.preview_png, hostOutputDir);
  return {
    ok: true,
    output_dcm: outputPayload,
    output_path: outputPayload?.selection?.path,
    preview_png: previewPayload,
    preview_path: previewPayload?.selection?.path || '',
  };
}

export default {
  meta: {
    name: 'T1rho',
    description: 'Fit a T1rho map from four MRI DICOM instances and display a jet pseudo-color preview.',
  },
  async setup(ctx = {}, electronApp) {
    return electronApp.whenReady().then(() => init(ctx));
  },
  ui: {
    entry: 'ui/index.html',
    windowOptions: {
      width: 640,
      height: 900,
      minWidth: 520,
      minHeight: 680,
    },
  },
  api: {
    async fit_t1rho_map(instance1, tsl1_ms, instance2, tsl2_ms, instance3, tsl3_ms, instance4, tsl4_ms, outputDir) {
      return fitT1rhoMap(instance1, tsl1_ms, instance2, tsl2_ms, instance3, tsl3_ms, instance4, tsl4_ms, outputDir);
    },
  },
};