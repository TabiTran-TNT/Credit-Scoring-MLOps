"""Chunked offline scoring with atomic output and the same pipeline as the API."""
import argparse
import os
from pathlib import Path
from uuid import uuid4
import pandas as pd
from app.model_loader import get_feature_columns, get_model, get_model_id, get_model_version, load_model
from credit_scoring.scoring import score_frame
from monitoring.logger import log_event


def batch_score(input_path, output_path, chunk_size=1000):
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    source, destination = Path(input_path).resolve(), Path(output_path).resolve()
    if source == destination:
        raise ValueError('Input and output paths must differ')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.' + uuid4().hex + '.tmp')
    load_model()
    count = 0
    try:
        for chunk in pd.read_csv(source, chunksize=chunk_size):
            if chunk.empty:
                continue
            result = score_frame(chunk, get_model(), get_feature_columns(), get_model_id(), get_model_version())
            result.to_csv(temporary, mode='a', index=False, header=count == 0)
            count += len(result)
        if not count:
            raise ValueError('No rows to score')
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    log_event('batch_scoring', rows=count, model_id=get_model_id(), model_version=get_model_version())
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--chunk-size', type=int, default=1000)
    args = parser.parse_args()
    print(f'Scored {batch_score(args.input, args.output, args.chunk_size)} rows')
