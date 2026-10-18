#!/bin/sh
# The synthetic workspace is built into the image. The review store is the PostgreSQL service, so
# the alert queue is stored at start-up (its migrations run first), then the API is served.
set -eu
strata anomaly run --budget 50
exec python -c "from strata.api.app import main; main()"
