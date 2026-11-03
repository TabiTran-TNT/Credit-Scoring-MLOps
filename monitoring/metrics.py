from prometheus_client import Counter, Histogram, CONTENT_TYPE_LATEST, generate_latest

REQUESTS = Counter('credit_scoring_requests_total', 'Prediction responses', ['status'])
LATENCY = Histogram('credit_scoring_handler_seconds', 'Prediction handler elapsed seconds',
                    buckets=(.005, .01, .025, .05, .1, .25, .5, 1, 2, 5))
