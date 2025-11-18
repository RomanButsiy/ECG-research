from loguru import logger
from my_helpers.data_preparation import DataPreparation
from my_helpers.read_data.read_data_file import ReadDataFile
import pandas as pd
import neurokit2 as nk
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import time

class GenerateRhythmFunction(ReadDataFile):

    def __init__(self, ecg_config):
        super().__init__(ecg_config)

    def genFr(self):
        logger.info("Get ECG Peaks")
        path = f'{self.ecg_config.getImgPath()}/{self.ecg_config.getConfigBlock()}'
        path_all = f'{path}/FR-{self.ecg_config.getConfigBlock()}.csv'
        Path(path).mkdir(parents=True, exist_ok=True)
        if Path(path_all).is_file():
            logger.warning(f'File {path_all} is exist. Create a backup and continue')
            return
        i = self.ecg_config.getSigName()
        # lstart = time.time()
        _, rpeaks = nk.ecg_peaks(self.signals[i], sampling_rate=self.sampling_rate)
        _, waves = nk.ecg_delineate(self.signals[i], rpeaks, sampling_rate=self.sampling_rate)
        # lend = time.time()
        # print((lend-lstart)*10**3)
        ECG_P_Peaks = list(np.array(waves["ECG_P_Peaks"]) / self.sampling_rate)
        # ECG_Q_Peaks = list(np.round(np.array(waves["ECG_Q_Peaks"]) / self.sampling_rate, 4))
        ECG_R_Peaks = list(np.array(rpeaks["ECG_R_Peaks"]) / self.sampling_rate)
        # ECG_S_Peaks = list(np.round(np.array(waves["ECG_S_Peaks"]) / self.sampling_rate, 4))
        ECG_T_Peaks = list(np.array(waves["ECG_T_Peaks"]) / self.sampling_rate)     

        ecg_fr = pd.DataFrame({"ECG_P_Peaks" : ECG_P_Peaks, "ECG_R_Peaks" : ECG_R_Peaks, "ECG_T_Peaks" : ECG_T_Peaks})
        ecg_fr.interpolate(method='linear', inplace=True)
        nk.write_csv(ecg_fr.round(4), path_all)

    def getRFFunctionValue(self):
        logger.info("Get Rhythm Function Function Value")
        path = f'{self.ecg_config.getImgPath()}/{self.ecg_config.getConfigBlock()}'
        path_all = f'{path}/Ch-FR-{self.ecg_config.getConfigBlock()}.csv'
        ch = self.ecg_config.getSigName()
        new_val = [f"Ch_{ch}_ECG_P_Peaks", f"Ch_{ch}_ECG_R_Peaks", f"Ch_{ch}_ECG_T_Peaks"]
        self.getData(None)
        res = []
        for i in range(len(self.matrix_P_R)):
            res.append([self.matrix_P_R[i][0], self.matrix_R_T[i][0], self.matrix_T_P[i][0]])
        res.append([None, None, None])
        df = pd.read_csv(path_all, index_col="Unnamed: 0").round(4)
        df[new_val] = res
        df.to_csv(f'{path}/Ch-FR-{self.ecg_config.getConfigBlock()}.csv')

    def plotIntervals(self):
        logger.info("Plot ECG Intervals")

        path = f'{self.ecg_config.getImgPath()}/{self.ecg_config.getConfigBlock()}'
        intervals_path = f'{path}/FR-{self.ecg_config.getConfigBlock()}.csv'

        if not Path(intervals_path).is_file():
            e = 'The rhythm function file %s does not exist' % intervals_path
            logger.error(e)
            raise FileNotFoundError(e)

        # data = DataPreparation(self.ecg_config, None)
        # ECG_data = np.concatenate(data.getPreparedData())
        
        ECG_data = (self.signals[self.ecg_config.getSigName()])
        time = np.arange(0, len(ECG_data), 1) / self.sampling_rate

        intervals_fr = pd.read_csv(intervals_path)
        # ECG_P_Onsets = intervals_fr["ECG_P_Onsets"]
        ECG_P_Peaks = intervals_fr["ECG_P_Peaks"]
        # ECG_P_Offsets = intervals_fr["ECG_P_Offsets"]
        # ECG_Q_Peaks = intervals_fr["ECG_Q_Peaks"]
        ECG_R_Peaks = intervals_fr["ECG_R_Peaks"]
        # ECG_S_Peaks = intervals_fr["ECG_S_Peaks"]
        # ECG_T_Onsets = intervals_fr["ECG_T_Onsets"]
        ECG_T_Peaks = intervals_fr["ECG_T_Peaks"]
        # ECG_T_Offsets = intervals_fr["ECG_T_Offsets"]

        plt.clf()
        plt.rcParams.update({'font.size': 20})
        f, axis = plt.subplots()
        f.set_size_inches(19, 6)
        f.tight_layout()
        axis.grid(True)
        # signal_detrended = detrend(ECG_data)
        # print(len(signal_detrended))
        axis.plot(time, ECG_data, linewidth=3, label=r"$\xi_{{\omega}} (t), NU$")
        # axis.plot(time, ECG_data, linewidth=3, label=r"$\xi_{{\omega}} (t), mV$")
        for t in range(len(ECG_P_Peaks) - 1):

            i = t
            P_Peaks = ECG_P_Peaks[i] 
            # Q_Peaks = ECG_Q_Peaks[i] + T
            R_Peaks = ECG_R_Peaks[i] 
            # S_Peaks = ECG_S_Peaks[i] + T
            T_Peaks = ECG_T_Peaks[i]
            next_P_Peaks = ECG_P_Peaks[i + 1]

            # axis.axvline(x = P_Onsets, color = '#9467bd')
            axis.axvline(P_Peaks, color = '#d62728', linewidth=4)
            # axis.axvline(x = P_Peaks, color = '#ff7f0e')
            # axis.axvline(x = P_Offsets, color = '#8c564b')
            # axis.axvline(x = Q_Peaks, color = '#e377c2')
            # axis.axvspan(P_Peaks, R_Peaks, color = '#9467bd', alpha=0.2)
            # axis.axvline(x = S_Peaks, color = '#7f7f7f')
            # axis.axvline(x = T_Onsets, color = '#bcbd22')
            axis.axvspan(P_Peaks, T_Peaks, color = '#ff7f0e', alpha=0.2)
            axis.axvspan(T_Peaks, next_P_Peaks, color = '#2ca02c', alpha=0.2)
            # axis.axvline(x = T_Peaks, color = '#d62728')
            # axis.axvline(x = T_Offsets, color = '#17becf')
        axis.set_xlabel("$t, s$", loc = 'right')
        axis.legend(loc='upper right')
        # axis.axis(ymin = -300, ymax = 300)
        # axis.axis(xmin = 0.5, xmax = 5.2)
        plt.show()
        # plt.savefig(f'{path}/ecg_{self.ecg_config.getSigName()}.png', dpi=300)
        

    def showIntervals(self):
        logger.info("Show ECG Intervals")
        # path = f'{self.ecg_config.getImgPath()}/{self.ecg_config.getConfigBlock()}/Intervals'
        # intervals_path = f'{path}/All.csv'

        path = f'{self.ecg_config.getImgPath()}/{self.ecg_config.getConfigBlock()}'
        intervals_path = f'{path}/FR-{self.ecg_config.getConfigBlock()}.csv'

        if not Path(intervals_path).is_file():
            e = 'The rhythm function file %s does not exist' % intervals_path
            logger.error(e)
            raise FileNotFoundError(e)

        # data = DataPreparation(self.ecg_config, None)
        # ECG_data = np.concatenate(data.getPreparedData())
        
        ECG_data = (self.signals[self.ecg_config.getSigName()])
        time = np.arange(0, len(ECG_data), 1) / self.sampling_rate

        intervals_fr = pd.read_csv(intervals_path)
        # ECG_P_Onsets = intervals_fr["ECG_P_Onsets"]
        ECG_P_Peaks = intervals_fr["ECG_P_Peaks"]
        # ECG_P_Offsets = intervals_fr["ECG_P_Offsets"]
        # ECG_Q_Peaks = intervals_fr["ECG_Q_Peaks"]
        ECG_R_Peaks = intervals_fr["ECG_R_Peaks"]
        # ECG_S_Peaks = intervals_fr["ECG_S_Peaks"]
        # ECG_T_Onsets = intervals_fr["ECG_T_Onsets"]
        ECG_T_Peaks = intervals_fr["ECG_T_Peaks"]
        # ECG_T_Offsets = intervals_fr["ECG_T_Offsets"]

        # plt.clf()
        # plt.rcParams.update({'font.size': 20})
        # f, axis = plt.subplots()
        # f.set_size_inches(19, 6)
        # f.tight_layout()
        # axis.grid(True)
        # signal_detrended = detrend(ECG_data)
        # print(len(signal_detrended))
        # print(len(ECG_data))

        # axis.plot(time, ECG_data, linewidth=3, label=r"$\xi_{{\omega}} (t), NU$")
        # axis.plot(time, ECG_data, linewidth=3, label=r"$\xi_{{\omega}} (t), mV$")
        # for P_Peaks, R_Peaks, T_Peaks in zip(ECG_P_Peaks, ECG_R_Peaks, ECG_T_Peaks):
        #     # axis.axvline(x = P_Onsets, color = '#9467bd')
        #     # axis.axvline(x = P_Peaks, color = '#d62728', linewidth=4)
        #     axis.axvline(x = P_Peaks, color = '#ff7f0e')
        #     # axis.axvline(x = P_Offsets, color = '#8c564b')
        #     # axis.axvline(x = Q_Peaks, color = '#e377c2')
        #     axis.axvline(x = R_Peaks, color = '#d62728')
        #     # axis.axvline(x = S_Peaks, color = '#7f7f7f')
        #     # axis.axvline(x = T_Onsets, color = '#bcbd22')
        #     axis.axvline(x = T_Peaks, color = '#2ca02c')
        #     # axis.axvline(x = T_Peaks, color = '#d62728')
        #     # axis.axvline(x = T_Offsets, color = '#17becf')
        # axis.set_xlabel("$t, s$", loc = 'right')
        # axis.legend(loc='upper right')
        # axis.legend(['$T(t, 1), s$'])
        # axis.axis(ymin = -300, ymax = 300)
        
        # plt.show()
        # axis.axis(xmin = 0, xmax = 10)
        # plt.savefig(f'{path}/l-ecg_{self.ecg_config.getSigName()}.png', dpi=300)

        # ECG_data = np.concatenate(data.getPreparedData())
        
        # # ECG_data = (self.signals[self.ecg_config.getSigName()])
        # time = np.arange(0, len(ECG_data), 1) / self.sampling_rate

        # intervals_fr = pd.read_csv(intervals_path)
        # # ECG_P_Onsets = intervals_fr["ECG_P_Onsets"]
        # ECG_P_Peaks = intervals_fr["ECG_P_Peaks"]
        # # ECG_P_Offsets = intervals_fr["ECG_P_Offsets"]
        # # ECG_Q_Peaks = intervals_fr["ECG_Q_Peaks"]
        # ECG_R_Peaks = intervals_fr["ECG_R_Peaks"]
        # # ECG_S_Peaks = intervals_fr["ECG_S_Peaks"]
        # # ECG_T_Onsets = intervals_fr["ECG_T_Onsets"]
        # ECG_T_Peaks = intervals_fr["ECG_T_Peaks"]
        # # ECG_T_Offsets = intervals_fr["ECG_T_Offsets"]

        plt.clf()
        plt.rcParams.update({'font.size': 21})
        f, axis = plt.subplots()
        f.set_size_inches(16, 6)
        axis.grid(True)
        # signal_detrended = detrend(ECG_data)
        # print(len(signal_detrended))
        # print(len(ECG_data))
        axis.plot(time, ECG_data, linewidth=3, label=r"$\xi_{{\omega}} (t), NU$")
        # axis.plot(time, ECG_data, linewidth=3, label=r"$\xi_{{\omega}} (t), mV$")
        # i = 0
        for t in range(len(ECG_P_Peaks) - 1):

            i = t
            # T = (ECG_P_Peaks[i + 1] - ECG_P_Peaks[i])
            T = 0
            P_Peaks = ECG_P_Peaks[i] + T
            # Q_Peaks = ECG_Q_Peaks[i] + T
            R_Peaks = ECG_R_Peaks[i] + T
            # S_Peaks = ECG_S_Peaks[i] + T
            T_Peaks = ECG_T_Peaks[i] + T
            next_P_Peaks = ECG_P_Peaks[i + 1] + T

            # axis.axvline(x = P_Onsets, color = '#9467bd')
            axis.axvline(P_Peaks, color = '#d62728', linewidth=4)
            # axis.axvline(x = P_Peaks, color = '#ff7f0e')
            # axis.axvline(x = P_Offsets, color = '#8c564b')
            # axis.axvline(x = Q_Peaks, color = '#e377c2')
            # axis.axvspan(P_Peaks, R_Peaks, color = '#9467bd', alpha=0.2)
            # axis.axvline(x = S_Peaks, color = '#7f7f7f')
            # axis.axvline(x = T_Onsets, color = '#bcbd22')
            # axis.axvspan(R_Peaks, T_Peaks, color = '#ff7f0e', alpha=0.2)
            # axis.axvspan(R_Peaks, T_Peaks, color = '#ff7f0e', alpha=0.2)
            # axis.axvspan(T_Peaks, next_P_Peaks, color = '#2ca02c', alpha=0.2)
            # axis.axvline(x = T_Peaks, color = '#d62728')
            # axis.axvline(x = T_Offsets, color = '#17becf')
            # i = i + 1
        axis.set_xlabel("$t, Hour$", loc = 'right')
        axis.legend(loc='upper right')
        # axis.legend(['$T(t, 1), s$'])
        # axis.axis(ymin = -4, ymax = 5)
        # axis.axis(xmin = ECG_P_Peaks[0] - 0.005, xmax = 5.7)
        axis.axis(xmin = 1, xmax = 100)
        # f.tight_layout()
        # plt.show()
        plt.savefig(f'{path}/ecg_{self.ecg_config.getSigName()}.png', dpi=300)

    def getIntervals(self):
        logger.info("To file ECG Intervals") 
        path = f'{self.ecg_config.getImgPath()}/{self.ecg_config.getConfigBlock()}/Intervals'
        intervals_path = f'{path}/All.csv'
        if not Path(intervals_path).is_file():
            e = 'The rhythm function file %s does not exist' % intervals_path
            logger.error(e)
            raise FileNotFoundError(e)
        
        intervals_fr = pd.read_csv(intervals_path)
        ECG_P_Onsets = intervals_fr["ECG_P_Onsets"]
        ECG_P_Offsets = intervals_fr["ECG_P_Offsets"]
        ECG_Q_Peaks = intervals_fr["ECG_Q_Peaks"]
        ECG_S_Peaks = intervals_fr["ECG_S_Peaks"]
        ECG_T_Onsets = intervals_fr["ECG_T_Onsets"]
        ECG_T_Offsets = intervals_fr["ECG_T_Offsets"]

        P = ECG_P_Offsets - ECG_P_Onsets
        P_df = pd.DataFrame({"Y" : round(P, 4)})
        P_df.index = np.arange(1, len(P_df) + 1)
        P_df.index.name = "X"
        nk.write_csv(P_df, f'{path}/P.csv')

        PQ = ECG_Q_Peaks - ECG_P_Offsets
        PQ_df = pd.DataFrame({"Y" : round(PQ, 4)})
        PQ_df.index = np.arange(1, len(PQ_df) + 1)
        PQ_df.index.name = "X"
        nk.write_csv(PQ_df, f'{path}/pq.csv')

        QRS = ECG_S_Peaks - ECG_Q_Peaks
        QRS_df = pd.DataFrame({"Y" : round(QRS, 4)})
        QRS_df.index = np.arange(1, len(QRS_df) + 1)
        QRS_df.index.name = "X"
        nk.write_csv(QRS_df, f'{path}/qRs.csv')

        T = ECG_T_Offsets - ECG_T_Onsets
        T_df = pd.DataFrame({"Y" : round(T, 4)})
        T_df.index = np.arange(1, len(T_df) + 1)
        T_df.index.name = "X"
        nk.write_csv(T_df, f'{path}/T.csv')

        ECG_P_Onsets_arr = ECG_P_Onsets.to_numpy()
        ECG_T_Offsets_arr = ECG_T_Offsets.to_numpy()

        TP = ECG_P_Onsets_arr[1:] - ECG_T_Offsets_arr[:-1]
        TP_df = pd.DataFrame({"Y" : np.round(TP, 4)})
        TP_df.index = np.arange(1, len(TP_df) + 1)
        TP_df.index.name = "X"
        nk.write_csv(TP_df, f'{path}/tp.csv')


    def plotECG(self):
        logger.info("Plot ECG")
        path = f'{self.ecg_config.getImgPath()}/{self.ecg_config.getConfigBlock()}'
        Path(path).mkdir(parents=True, exist_ok=True)

        data = DataPreparation(self.ecg_config, None)

        # print(data.getPreparedData())

        # ECG_data = np.transpose(data.getPreparedData()[10])

        # ECG_data = [*ECG_data]

        # ECG_data = np.mean(data.getPreparedData(), 0)

        # ECG_data = [*ECG_data]* 6


        start = 0
        end = 5000
        ECG_data = np.round(self.signals[self.ecg_config.getSigName()], 4)
        time = np.arange(0, len(ECG_data), 1) / self.sampling_rate

        # data = pd.DataFrame({"ECG_data" : ECG_data})
        # data.index = time
        # data.index.name = "Time"
        # nk.write_csv(data, f'{path}/ECG_data.csv')

        plt.clf()
        plt.rcParams.update({'font.size': 15})
        f, axis = plt.subplots(1)
        f.tight_layout()
        f.set_size_inches(19, 6)
        axis.grid(True)
        axis.plot(time, ECG_data, linewidth=3, label=r"$\xi_{{\omega}} (t), mV$")
        # for i in range(6):
        #     axis.axvline(x = i, color = '#d62728', linewidth=4)
        axis.set_xlabel("$t, s$", loc = 'right')
        axis.legend(loc='upper right')
        # axis.axis(ymin = -6, ymax = 6)
        axis.axis(xmin = 0.5, xmax = 5.7)
        plt.savefig(f'{path}/ECG_data_Chanel_{self.ecg_config.getSigName()}.png', dpi=300)


    def plotFr(self, debug = False):
        logger.info("Plot a rhythm function")

        self.getData(None)

        plot_path = f'{self.ecg_config.getFrImgPath()}/{self.ecg_config.getConfigBlock()}'
        Path(plot_path).mkdir(parents=True, exist_ok=True)

        T1_ECG_T_Peaks = []
        T1_ECG_P_Peaks = []
        T1_ECG_R_Peaks = []
        T1_ECG_S_Peaks = []
        T1_ECG_Q_Peaks = []
        T1_X = []
        T1_Y = []

        for i in range(len(self.ECG_T_Peaks)-1):
            T1_ECG_T_Peaks.append(round(self.ECG_T_Peaks[i+1] - self.ECG_T_Peaks[i], 2))

        for i in range(len(self.ECG_P_Peaks)-1):
            T1_ECG_P_Peaks.append(round(self.ECG_P_Peaks[i+1] - self.ECG_P_Peaks[i], 2))

        for i in range(len(self.ECG_R_Peaks)-1):
            T1_ECG_R_Peaks.append(round(self.ECG_R_Peaks[i+1] - self.ECG_R_Peaks[i], 2))

        # if self.Q_S_exist:
        #     for i in range(len(self.ECG_S_Peaks)-1):
        #         T1_ECG_S_Peaks.append(round(self.ECG_S_Peaks[i+1] - self.ECG_S_Peaks[i], 2))

        #     for i in range(len(self.ECG_Q_Peaks)-1):
        #         T1_ECG_Q_Peaks.append(round(self.ECG_Q_Peaks[i+1] - self.ECG_Q_Peaks[i], 2))

        for i in range(len(self.ECG_P_Peaks) - 1):
            T1_X.append(self.ECG_P_Peaks[i])
            # if i == B_i :
            #     break
            # if self.Q_S_exist:
            #     T1_X.append(self.ECG_Q_Peaks[i])
            T1_X.append(self.ECG_R_Peaks[i])
            # if self.Q_S_exist:
            #     T1_X.append(self.ECG_S_Peaks[i])
            T1_X.append(self.ECG_T_Peaks[i])

        for i in range(len(T1_ECG_P_Peaks)):
            T1_Y.append(T1_ECG_P_Peaks[i])
            # if i == B_i :
            #     break
            # if self.Q_S_exist:
            #     T1_Y.append(T1_ECG_Q_Peaks[i])
            T1_Y.append(T1_ECG_R_Peaks[i])
            # if self.Q_S_exist:
            #     T1_Y.append(T1_ECG_S_Peaks[i])
            T1_Y.append(T1_ECG_T_Peaks[i])

        # if self.Q_S_exist:
            # fr = pd.DataFrame({"FR_ECG_P" : T1_ECG_P_Peaks, "FR_ECG_Q" : T1_ECG_Q_Peaks, "FR_ECG_R" : T1_ECG_R_Peaks, "FR_ECG_S" : T1_ECG_S_Peaks, "FR_ECG_T" : T1_ECG_T_Peaks})
            # nk.write_csv(fr, f'{plot_path}/FR.csv')
        print("Mathematical expectation")
        m = np.mean(T1_Y)
        print("%.5f" % m)
        print("Variance")
        v = sum((T1_Y - m)**2) / len(T1_Y)
        print("%.5f" % v)

        # T1_Y = list(map(lambda x: x - np.mean(T1_Y), T1_Y))

        ymax = 0.1
        max_value = np.nanmax(T1_Y) if not np.isnan(np.nanmax(T1_Y)) else 0
        if ymax < max_value and not np.isinf(max_value):
                ymax = max_value + (max_value / 11.0)

        plt.clf()
        plt.rcParams.update({'font.size': 21})
        f, axis = plt.subplots(1)
        f.set_size_inches(16, 6)
        f.tight_layout()
        axis.grid(True)
        axis.plot(T1_X, T1_Y, linewidth=3)
        axis.set_xlabel("$t, s$", loc = 'right')
        axis.legend(['$T(t, 1), s$'])
        axis.axis(ymin = 0, ymax = ymax)
        axis.axis(xmin = T1_X[0], xmax = 100)
        plt.savefig(f'{plot_path}/FR-{self.ecg_config.getConfigBlock()}.png', dpi=300)
        return T1_X, T1_Y, plot_path

