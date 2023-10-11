import threading

from event_service_utils.logging.decorators import timer_logger
from event_service_utils.services.event_driven import BaseEventDrivenCMDService
from event_service_utils.tracing.jaeger import init_tracer

from semantic_image_segmentation.models.deeplabv3 import DeepLabv3ModelLoader


class SemanticImageSegmentation(BaseEventDrivenCMDService):
    def __init__(self,
                 service_stream_key, service_cmd_key_list,
                 pub_event_list, service_details,
                 model_base_configs,
                 stream_factory,
                 logging_level,
                 tracer_configs):
        tracer = init_tracer(self.__class__.__name__, **tracer_configs)
        super(SemanticImageSegmentation, self).__init__(
            name=self.__class__.__name__,
            service_stream_key=service_stream_key,
            service_cmd_key_list=service_cmd_key_list,
            pub_event_list=pub_event_list,
            service_details=service_details,
            stream_factory=stream_factory,
            logging_level=logging_level,
            tracer=tracer,
        )
        self.cmd_validation_fields = ['id']
        self.data_validation_fields = ['id']

        self.model_base_configs = model_base_configs
        if self.model_base_configs.get('setup_model_on_start', True):
            self.setup_model_loader()

    # def publish_some_event_type(self, event_data):
    #     self.publish_event_type_to_stream(event_type=PUB_EVENT_TYPE_SOME_EVENT_TYPE, new_event_data=event_data)

    def setup_model_loader(self):
        self.model_loader = DeepLabv3ModelLoader(
            self.model_base_configs
        )

    @timer_logger
    def process_data_event(self, event_data, json_msg):
        if not super(SemanticImageSegmentation, self).process_data_event(event_data, json_msg):
            return False
        # do something here
        pass

    def process_event_type(self, event_type, event_data, json_msg):
        if not super(SemanticImageSegmentation, self).process_event_type(event_type, event_data, json_msg):
            return False
        if event_type == 'SomeEventType':
            # do some processing
            pass
        elif event_type == 'OtherEventType':
            # do some other processing
            pass

    def log_state(self):
        super(SemanticImageSegmentation, self).log_state()
        self.logger.info(f'Service name: {self.name}')
        # function for simple logging of python dictionary
        # self._log_dict('Some Dictionary', self.some_dict)

    def run(self):
        super(SemanticImageSegmentation, self).run()
        self.log_state()
        # self.cmd_thread = threading.Thread(target=self.run_forever, args=(self.process_cmd,))
        self.data_thread = threading.Thread(target=self.run_forever, args=(self.process_data,))
        # self.cmd_thread.start()
        self.data_thread.start()
        # self.cmd_thread.join()
        self.data_thread.join()
