"""Profile inference, export ONNX and require numerical/decision parity."""
import argparse
import cProfile
import io
import json
from pathlib import Path
import pstats
from time import perf_counter

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from credit_scoring.artifacts import atomic_json
from credit_scoring.tracking import hash_file


def benchmark(model_path, validation_path, output_dir, repeats=200):
    if repeats < 5:
        raise ValueError('At least five repetitions are required')
    model = joblib.load(model_path)
    X = pd.read_csv(validation_path)
    if list(X.columns) != list(model.feature_names_in_) or X.empty:
        raise ValueError('Validation input must match the model feature schema')
    converted = convert_sklearn(
        model, initial_types=[('features', FloatTensorType([None, X.shape[1]]))],
        options={id(model.named_steps['classifier']): {'zipmap': False}}, target_opset=17,
    )
    onnx.checker.check_model(converted)
    session = ort.InferenceSession(converted.SerializeToString(), providers=['CPUExecutionProvider'])
    arrays = X.to_numpy(dtype=np.float32)
    # Check all supplied rows, including missing values, rather than one example.
    expected = model.predict_proba(X)
    actual = session.run(None, {'features': arrays})[1]
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(actual[:, 1] >= .5, expected[:, 1] >= .5)
    row, array = X.iloc[:1], arrays[:1]
    functions = {'sklearn': lambda: model.predict_proba(row),
                 'onnx': lambda: session.run(None, {'features': array})}
    timings = {}
    for name, function in functions.items():
        for _ in range(10):
            function()
        times = []
        for _ in range(repeats):
            started = perf_counter()
            function()
            times.append((perf_counter() - started) * 1000)
        timings[name] = {'mean_ms': float(np.mean(times)), 'p95_ms': float(np.percentile(times, 95))}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    profiler = cProfile.Profile()
    profiler.runcall(model.predict_proba, X.head(100))
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats('cumulative').print_stats(30)
    (output / 'profile.txt').write_text(stream.getvalue(), encoding='utf-8')
    (output / 'model.onnx').write_bytes(converted.SerializeToString())
    report = {'model_sha256': hash_file(model_path), 'validation_sha256': hash_file(validation_path),
              'parity_rows': len(X), 'decision_parity': True, 'rtol': 1e-4, 'atol': 1e-5,
              'timings': timings, 'scope': 'single-row model inference; excludes HTTP and raw preprocessing',
              'speedup': timings['sklearn']['mean_ms'] / timings['onnx']['mean_ms']}
    atomic_json(output / 'benchmark.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--validation', required=True)
    parser.add_argument('--output', default='reports/optimization')
    parser.add_argument('--repeats', type=int, default=200)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.model, args.validation, args.output, args.repeats), indent=2))
