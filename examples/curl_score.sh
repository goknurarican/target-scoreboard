#!/bin/bash
curl -s -X POST http://localhost:8000/score \
  -H 'Content-Type: application/json' \
  -d @examples/sample_payload.json | jq .
