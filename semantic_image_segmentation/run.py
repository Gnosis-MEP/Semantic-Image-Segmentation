#!/usr/bin/env python
from event_service_utils.streams.redis import RedisStreamFactory

from semantic_image_segmentation.service import SemanticImageSegmentation

from semantic_image_segmentation.conf import (
    REDIS_ADDRESS,
    REDIS_PORT,
    PUB_EVENT_LIST,
    SERVICE_STREAM_KEY,
    SERVICE_CMD_KEY_LIST,
    LOGGING_LEVEL,
    TRACER_REPORTING_HOST,
    TRACER_REPORTING_PORT,
    SERVICE_DETAILS,
    MODEL_NAME,
    SETUP_MODEL_ON_START,
)


def run_service():
    tracer_configs = {
        'reporting_host': TRACER_REPORTING_HOST,
        'reporting_port': TRACER_REPORTING_PORT,
    }
    stream_factory = RedisStreamFactory(host=REDIS_ADDRESS, port=REDIS_PORT)

    model_base_configs = {
        'model_name': MODEL_NAME,
        'hot_start': True,
        'setup_model_on_start': SETUP_MODEL_ON_START,
    }

    service = SemanticImageSegmentation(
        service_stream_key=SERVICE_STREAM_KEY,
        service_cmd_key_list=SERVICE_CMD_KEY_LIST,
        pub_event_list=PUB_EVENT_LIST,
        service_details=SERVICE_DETAILS,
        model_base_configs=model_base_configs,
        stream_factory=stream_factory,
        logging_level=LOGGING_LEVEL,
        tracer_configs=tracer_configs
    )
    service.run()


def main():
    try:
        run_service()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
