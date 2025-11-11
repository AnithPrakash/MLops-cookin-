#!/usr/bin/env bash

cd "$(dirname "$0")"


LOCAL_TAG=`date +"%Y-%m-%d-%H-%M"`
export LOCAL_IMAGE_NAME="stream-model-duration:${LOCAL_TAG}"

docker build -t ${LOCAL_IMAGE_NAME} .

docker-compose up -d

sleep 1

# docker run -it --rm \
#     -p 8080:8080 \
#     -e PREDICTIONS_STREAM_NAME="ride_predictions" \
#     -e RUN_ID="Test123" \
#     -e MODEL_LOCATION="/app/model/" \
#     -e TEST_RUN="True" \
#     -e AWS_DEFAULT_REGION="eu-west-1" \
#     -v /workspaces/MLops-cookin-/06-Best_practices/integration-test/model:/app/model \
#     stream-model-duration:v2

pipenv run python test_docker.py