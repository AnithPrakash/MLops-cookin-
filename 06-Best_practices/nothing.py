if not TEST_RUN:
    kinesis_client.put_record(
        StreamName=PREDICTIONS_STREAM_NAME,
        Data=json.dumps(prediction_event),
        PartitionKey=str(ride_id)
    )