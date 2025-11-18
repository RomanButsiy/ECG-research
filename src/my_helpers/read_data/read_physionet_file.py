from loguru import logger
import numpy as np
import wfdb

class ReadPhysionetFile:

    def __init__(self, ecg_config):
        data_path = f'{ecg_config.getFileName()}'
        logger.info("Read physionet file")
        logger.info(data_path)

        signals, self.fileds = wfdb.rdsamp(data_path)

        self.sampling_rate = self.fileds['fs']
        self.signals = signals.transpose()

        bandpass_notch_channels = []
        for i in self.signals:
            bandpass_notch_channels.append(self.bandpass(i, fs = self.sampling_rate))

        self.signals = bandpass_notch_channels

        logger.info(f'Fileds: {self.fileds["sig_name"]}')
        logger.info(f'Sampling rate: {self.sampling_rate}')


    def bandpass(self, data, fs):
        # data = -data
        m = np.mean(data)
        data = (data - m)
        t = 1
        if np.max(data) > 1000:
            t = 1000.0
        res = data / t
        return res
        # return res / 1000.0
    

    # def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
    #     b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
    #     y = lfilter(b, a, data)
    #     return y
    
    # def butter_bandpass(self, lowcut, highcut, fs, order=5):
    #     nyq = 0.5 * fs
    #     low = lowcut / nyq
    #     high = highcut / nyq
    #     b, a = butter(order, [low, high], btype='band')
    #     return b, a
        