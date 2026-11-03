import argparse
import json
from credit_scoring.config import get_tracking_config
from credit_scoring.lifecycle import promote_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Promote a validated version to a pinned local release')
    parser.add_argument('--version', required=True)
    parser.add_argument('--name', default=get_tracking_config().registered_model_name)
    args = parser.parse_args()
    print(json.dumps(promote_model(args.name, args.version), indent=2))
