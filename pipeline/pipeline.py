import threading
from pipe_save import pipe  # 기존 pipe 함수가 정의된 파일

class DataPipeline:
    def __init__(self, device, seq_len=10):
        self.device = device
        self.seq_len = seq_len
        self.buffer = []
        self.data_buffer = {
            "timestamps": [],
            "io_types": [],
            "sectors": [],
            "sizes": [],
            "anomalies": []
        }

    def collect_data(self, stop_event, process_func):
        """
        Collect data using the `pipe` function and process it.
        Args:
            stop_event (threading.Event): To stop data collection.
            process_func (callable): Function to process each batch of data.
        """
        try:
            for raw_data in pipe(self.device, stop_event):
                self.buffer.append(raw_data)
                if len(self.buffer) > self.seq_len:
                    self.buffer.pop(0)  # Maintain sliding window
                if len(self.buffer) == self.seq_len:
                    process_func(self.buffer, self.data_buffer)
        except KeyboardInterrupt:
            stop_event.set()

    def start(self, process_func):
        """
        Start the data pipeline in a separate thread.
        Args:
            process_func (callable): Function to process collected data.
        """
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self.collect_data, args=(stop_event, process_func)
        )
        thread.start()
        return stop_event
