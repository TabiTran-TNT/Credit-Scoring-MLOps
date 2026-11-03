import argparse
import json
import pandas as pd
from app.model_loader import load_model, get_model, get_model_id
from monitoring.drift_report import build_report, log_window

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate feature/score PSI, quality and Evidently reports')
    parser.add_argument('--reference', required=True, help='Reference CSV from the serving model training run')
    parser.add_argument('--logs', default='logs/predictions.jsonl')
    parser.add_argument('--output', default='reports/monitoring')
    parser.add_argument('--since', help='Timezone-aware ISO timestamp')
    parser.add_argument('--max-rows', type=int, default=20000)
    parser.add_argument('--labels', help='Optional mature-label CSV keyed by request_id')
    args = parser.parse_args()
    load_model()
    rows, errors, malformed = log_window(args.logs, get_model_id(), args.since, args.max_rows)
    result = build_report(pd.read_csv(args.reference), rows, get_model(), get_model_id(),
                          args.output, errors=errors, malformed=malformed, labels=args.labels)
    print(json.dumps({'rows': result['rows'], 'alerts': result['alerts'], 'output': args.output}, indent=2))
    raise SystemExit(2 if result['action_required'] else 0)
