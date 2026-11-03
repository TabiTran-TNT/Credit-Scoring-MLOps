import argparse
import json
from credit_scoring.artifacts import ROOT
from credit_scoring.config import get_tracking_config
from credit_scoring.evaluation import load_rules, validate_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate a registry version on its recorded holdout')
    parser.add_argument('--version', required=True)
    parser.add_argument('--name', default=get_tracking_config().registered_model_name)
    parser.add_argument('--config', default=str(ROOT / 'configs/validation.yaml'))
    args = parser.parse_args()
    result = validate_model(args.name, args.version, load_rules(args.config))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
