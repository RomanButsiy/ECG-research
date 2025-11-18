from loguru import logger
import numpy as np
import pandas as pd

class ReadXLSFile:

    def __init__(self, ecg_config):
        data_path = f'{ecg_config.getFileName()}.{ecg_config.getDataType()}'
        logger.info("Read XLS file")
        logger.info(data_path)

        # columns = ["'Elapsed time'", "'I'", "'II'", "'ECG1'", "'ECG2'", "'III'", "'V'"]

        excel_data = pd.read_excel(data_path, engine='openpyxl')
        data = pd.DataFrame(excel_data)

        data = data.dropna(axis=1, how='all')
        data = data.dropna()

        self.signals = data.to_numpy().transpose()

        self.sampling_rate = round(1 / (self.signals[0][1] - self.signals[0][0]), 1)

        bandpass_notch_channels = []
        for i in self.signals:
            bandpass_notch_channels.append(self.bandpass(i, fs = self.sampling_rate))

        self.signals = bandpass_notch_channels

        print(self.signals)


        logger.info(f'Fileds: {data.columns.values}')
        logger.info(f'Sampling rate: {self.sampling_rate}')

        # cleaned_columns = [col.strip("'") for col in data.columns.values]

        # lines = []
        # counter = 1
        # for col in cleaned_columns:
        #     if col == "Elapsed time":
        #         continue
        #     lines.append(f"{counter} - {col}")
        #     counter += 1

        # with open(f"{ecg_config.getImgPath()}/{ecg_config.getConfigBlock()}/keys.txt", "w") as f:
        #     f.write("\n".join(lines))


    def bandpass(self, data, fs):
        # m = np.mean(data)
        # data = (data - m)
        # t = 1
        # if np.max(data) > 1000:
        #     t = 1000.0
        # res = data / t
        return data
        return res