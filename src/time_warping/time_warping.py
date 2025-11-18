from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
from my_helpers.read_data.read_data_file import ReadDataFile
from pathlib import Path
from loguru import logger
from matplotlib import cm
import time as tt_time

class TimeWarping(ReadDataFile):

    def __init__(self, ecg_config):
        super().__init__(ecg_config)

        self.base_path = f'{ecg_config.getImgPath()}/{ecg_config.getConfigBlock()}'
        self.tw_path = f'{self.base_path}/Time_Warping'
        Path(self.tw_path).mkdir(parents=True, exist_ok=True)
        self.fr_path = f'{self.base_path}/FR-{self.ecg_config.getConfigBlock()}.csv'

        if not Path(self.fr_path).is_file():
            e = 'The rhythm function file %s does not exist' % self.fr_path
            logger.error(e)
            raise FileNotFoundError(e)
        
        d = pd.read_csv(self.fr_path, usecols = ["ECG_P_Peaks", "ECG_R_Peaks", "ECG_T_Peaks"])
        d.rename(columns={'ECG_P_Peaks': 'd_m_0', 'ECG_R_Peaks': 'd_m_1', 'ECG_T_Peaks': 'd_m_2'}, inplace=True)
        d["d_m_3"] = d["d_m_0"].shift(-1)
        d = d[:-1] 
        t = pd.DataFrame(np.nan, index=np.arange(len(d)), columns=["t_m_0", "t_m_1", "t_m_2", "t_m_3"])
        z = pd.DataFrame(np.nan, index=np.arange(len(d)), columns=["z_m_0", "z_m_1", "z_m_2", "z_m_3"])
        c = pd.DataFrame(np.nan, index=np.arange(len(d)), columns=["c_m_1", "c_m_2", "c_m_3"])
        g = pd.DataFrame(np.nan, index=np.arange(len(d)), columns=["g_m_1", "g_m_2", "g_m_3"])
        a = pd.DataFrame(np.nan, index=np.arange(len(d)), columns=["a_m_1", "a_m_2", "a_m_3"])
        b = pd.DataFrame(np.nan, index=np.arange(len(d)), columns=["b_m_1", "b_m_2", "b_m_3"])
        i_data = pd.DataFrame(index=np.arange(len(d)), columns=["data_m_1", "data_m_2", "data_m_3"])
        y_time = pd.DataFrame(index=np.arange(len(d)), columns=["time_m_1", "time_m_2", "time_m_3"])
        o_data = pd.DataFrame(index=np.arange(len(d)), columns=["data_m_1", "data_m_2", "data_m_3"])
        o_time = pd.DataFrame(index=np.arange(len(d)), columns=["time_m_1", "time_m_2", "time_m_3"])
        alpha = pd.DataFrame(np.nan, index=np.arange(len(d)), columns=["a_m_1", "a_m_2", "a_m_3"])
        beta = pd.DataFrame(np.nan, index=np.arange(len(d)), columns=["b_m_1", "b_m_2", "b_m_3"])

        # Formula 28
        tw1_start = tt_time.time()
        T = d.iloc[0, -1] - d.iloc[0, 0]
        t.iloc[0] = np.array((d.iloc[0] - d.iloc[0, 0]))
        # t.iat[0, 0] = 0.0
        # t.iat[0, 1] = 0.1
        # t.iat[0, 2] = 0.69
        # t.iat[0, 3] = 0.697
        z.iloc[0] = np.array((d.iloc[0] - d.iloc[0, 0]) * ((2 * np.pi) / T))
        for i in t.columns:
            for m in t.index:
                t.at[m, i] = t.at[0, i] + m * T

        for i in z.columns:
            for m in z.index:
                z.at[m, i] = z.at[0, i] + m * 2 * np.pi
        
        for m in np.arange(0, t.shape[0]):
            for i in np.arange(1, t.shape[1]):
                c.iat[m, i-1] = (t.iat[m, i] - t.iat[m, i - 1]) / (d.iat[m, i] - d.iat[m, i-1])
                g.iat[m, i-1] = t.iat[m, i - 1] - (c.iat[m, i-1] * d.iat[m, i-1])
                a.iat[m, i-1] = 1.0 / c.iat[m, i-1]
                b.iat[m, i-1] = ((-1.0) * g.iat[m, i-1]) / c.iat[m, i-1]

                alpha.iat[m, i-1] = (z.iat[m, i] - z.iat[m, i - 1]) / (d.iat[m, i] - d.iat[m, i-1])
                beta.iat[m, i-1] = z.iat[m, i - 1] - (alpha.iat[m, i-1] * d.iat[m, i-1])

        time = np.arange(0, len(self.signals[self.ecg_config.getSigName()]), 1) / self.sampling_rate

        for m in np.arange(0, i_data.shape[0]):
            for i in np.arange(0, i_data.shape[1]):
                start = int(d.iat[m, i] * self.sampling_rate)
                end = int(d.iat[m, i + 1] * self.sampling_rate)
                tmp_data = np.array(self.signals[self.ecg_config.getSigName()][start:end])
                tmp_time = np.array(time[start:end])
                i_data.iat[m, i] = tmp_data
                o_time.iat[m, i] = ((c.iat[m, i] * tmp_time) + g.iat[m, i]) 
                if m == 0:
                    y_time.iat[m, i] = ((a.iat[m, i] * o_time.iat[m, i]) + b.iat[m, i])
                else:
                    y_time.iat[m, i] = ((a.iat[m, i] * (o_time.iat[0, i] + m * T)) + b.iat[m, i])

        for m in np.arange(0, i_data.shape[0]):
            for i in np.arange(0, i_data.shape[1]):
                if m == 0:
                    o_data.iat[m, i] = i_data.iat[m, i]
                    continue
                new_times = np.linspace(o_time.iat[m, i][0], o_time.iat[m, i][-1], len(o_time.iat[0, i]))
                interpolator = interp1d(o_time.iat[m, i], i_data.iat[m, i], kind='linear')
                o_data.iat[m, i] = interpolator(new_times)

        f_time = self.get_flatten(y_time)
        tw1_end = tt_time.time()
        tw1 = (tw1_end - tw1_start) + 0.7

        interp_matrix_all = self.get_main_data()
        self.i1_end = tt_time.time()
        i1 = (self.i1_end - self.i1_start) + 0.7

        # # Plot Autocovariation
        name = "Autocovariation function"
        # i5_start = tt_time.time()
        # i_c3 = self.getCorrelation(interp_matrix_all, correlation = True, deep = 3, multiply = True)
        # i5_end = tt_time.time()
        # self._3d_plot_to_file(i_c3, f_time, f"i {name} ecg {self.ecg_config.getSigName()}", size=(10, 10, 10), ztext=r'$\hat{C}_{2_{\xi}} (t_1, t_2), NU^2$')
        # tw5_start = tt_time.time()
        tw_c3 = self.getCorrelation(o_data, correlation = True, deep = 3, multiply = True)
        # tw5_end = tt_time.time()
        self._3d_plot_to_file(tw_c3, f_time, f"tw {name} ecg {self.ecg_config.getSigName()}", size=(10, 10, 10), ztext=r'$\hat{C}_{2_{\xi}} (t_1, t_2), NU^2$')
        # print(f"Root mean square distances: {np.sqrt(np.mean((np.concatenate(i_c3 - tw_c3)**2)))}")
        # self._3d_plot_to_file(i_c3 - tw_c3, f_time, f"diff {name} ecg {self.ecg_config.getSigName()}", size=(10, 10, 10), ztext=r'$\hat{C}_{2_{\xi}} (t_1, t_2), NU^2$')
        # Plot Autocorrelation
        name = "Autocorrelation function"
        # i5_start = tt_time.time()
        # i_c3 = self.getCorrelation(interp_matrix_all, correlation = False, deep = 3, multiply = True)
        # i5_end = tt_time.time()
        # self._3d_plot_to_file(i_c3, f_time, f"i {name} ecg {self.ecg_config.getSigName()}", size=(10, 10, 10), ztext=r'$\hat{R}_{2_{\xi}} (t_1, t_2), NU^2$')
        # tw5_start = tt_time.time()
        tw_c3 = self.getCorrelation(o_data, correlation = False, deep = 3, multiply = True)
        # tw5_end = tt_time.time()
        self._3d_plot_to_file(tw_c3, f_time, f"tw {name} ecg {self.ecg_config.getSigName()}", size=(10, 10, 10), ztext=r'$\hat{R}_{2_{\xi}} (t_1, t_2), NU^2$')
        # print(f"Root mean square distances: {np.sqrt(np.mean((np.concatenate(i_c3 - tw_c3)**2)))}")
        # self._3d_plot_to_file(i_c3 - tw_c3, f_time, f"diff {name} ecg {self.ecg_config.getSigName()}", size=(10, 10, 10), ztext=r'$\hat{R}_{2_{\xi}} (t_1, t_2), NU^2$')
        # i5 = i5_end - i5_start
        # tw5 = tw5_end - tw5_start
        # print(f"II {(i1 + i5)*10**3:.03f} ms")
        # print(f"TW {(tw1 + tw5)*10**3:.03f} ms")

        #-----------------------------------------------------------------------------------------------------------------------------------------
        # Mathematical statistics
        # self.plot_to_png(f"dd/3_tw_ecg_{self.ecg_config.getSigName()}", self.get_flatten(o_time), self.get_flatten(i_data), xlim=(0, 5.5), xtext="$t, s$", ytext=['$\\xi_{\\omega} (t), NU$'])
        # i2_start = tt_time.time()
        i_i_statistics = pd.concat([interp_matrix_all.agg([np.mean]), (interp_matrix_all**2).agg([np.mean]), (interp_matrix_all**3).agg([np.mean])], keys=[1, 2, 3])
        # i2_end = tt_time.time()
        # tw2_start = tt_time.time()
        tw_i_statistics = pd.concat([o_data.agg([np.mean]), (o_data**2).agg([np.mean]), (o_data**3).agg([np.mean])], keys=[1, 2, 3])
        # tw2_end = tt_time.time()
        # i3_start = tt_time.time()
        i_c_statistics = interp_matrix_all.agg(self.MyVar)
        # i3_end = tt_time.time()
        # tw3_start = tt_time.time()
        tw_c_statistics = o_data.agg(self.MyVar)
        # tw3_end = tt_time.time()
        # i2 = i2_end - i2_start
        # i3 = i3_end - i3_start
        # tw2 = tw2_end - tw2_start
        # tw3 = tw3_end - tw3_start


        self.getStatistics(i_i_statistics.loc[1, "mean"], tw_i_statistics.loc[1, "mean"], i_data, o_data, y_time, d, "1 Mathematical Expectation", [r"$\widehat{m}_{\xi_{\omega}} (t), NU$"], [r"$\Delta_{\widehat{m}_{\xi_{\omega}}} (t), NU$"])
        self.getStatistics(i_i_statistics.loc[2, "mean"], tw_i_statistics.loc[2, "mean"], i_data, o_data, y_time, d, "2 Initial Moments Second Order", [r"${\widehat{C}}_{\xi_{\omega}}^2 (t), NU^2$"], [r"$\Delta_{{\widehat{C}}_{\xi_{\omega}}^2} (t), NU^2$"])
        self.getStatistics(i_i_statistics.loc[3, "mean"], tw_i_statistics.loc[3, "mean"], i_data, o_data, y_time, d, "3 Initial Moments Third Order", [r"${\widehat{C}}_{\xi_{\omega}}^3 (t), NU^3$"], [r"$\Delta_{{\widehat{C}}_{\xi_{\omega}}^3} (t), NU^3$"])
        self.getStatistics(i_c_statistics, tw_c_statistics, i_data, o_data, y_time, d, "5 Variance", [r"${\widehat{d}}_{\xi_{\omega}}^2 (t), NU^2$"], [r"$\Delta_{{\widehat{d}}_{\xi_{\omega}}^2} (t), NU^2$"])
        # print(f"II {(i1 + i2 + self.i4)*10**3:.03f} ms")
        # print(f"TW {(tw1 + tw2 + self.tw4)*10**3:.03f} ms")
        
        #------------------------------------------------------------------------------------------------------------------------------------
        # y_2, y_1, x_1 = self.get_Psi(d, z, alpha, beta)
        # self.plot_to_png("2 Nu_1", np.concatenate(x_1), np.concatenate(y_2), xtext="$t, s$", ytext=['$\\widehat{\\nu}(t), s$'], ylim=(0, 10) ,xlim=(x_1[0][0], x_1[-1][-1]), piece = False)
        # self.plot_to_png("Psi_1", np.concatenate(x_1), np.concatenate(y_1), xtext="$t, s$", ytext=['$\\widehat{\\psi}^{-1}(t), s$'])
        # y_1, x_1 = self.getY_1(t, d, c, g)
        # self.plot_to_png("2 y_1", np.concatenate(x_1), np.concatenate(y_1), xtext="$t, s$", ytext=['$y^{-1}(t), s$'])
        # y_2, x_2 = self.getY_(t, a, b)
        # self.plot_to_png("2 y_", np.concatenate(x_2), np.concatenate(y_2), xtext="$t, s$", ytext=['$y(t), s$'])
        # self.plot_to_2png("y_y_1", np.concatenate(x_1), np.concatenate(y_1), np.concatenate(x_2), np.concatenate(y_2), xtext="$t, s$", ytext=['$y(t), s$', '$y^{-1}(t), s$'])
        # y_1, x_1 = self.getY_1(t, d, c, g, t__ = True)
        # self.plot_to_png("2 s_1", np.concatenate(x_1), np.concatenate(y_1), xtext="$t, s$", ytext=['$s^{-1}(t), s$'])
        # y_2, x_2 = self.getY_(t, a, b, t__ = True)
        # self.plot_to_png("2 s_", np.concatenate(x_1), np.concatenate(y_1), xtext="$t, s$", ytext=['$s(t), s$'])
        # self.plot_to_2png("y s_s_1", np.concatenate(x_1), np.concatenate(y_1), np.concatenate(x_2), np.concatenate(y_2), xtext="$t, s$", ytext=['$s(t), s$', '$s^{-1}(t), s$'])

    def getCorrelation(self, interp_matrix_all, correlation = False, deep = 1, multiply = False):
        tmp = None
        matrix_all = np.array(interp_matrix_all.apply(self.combine_columns, axis=1))
        m_ = np.mean(matrix_all)
        matrix_all_len = len(matrix_all)

        for i in range(matrix_all_len - deep + 1):
            concated = np.concatenate(matrix_all[i: i + deep])
            # print(np.concatenate(matrix_all[i: i + deep]))
            one = matrix_all[i]
            if correlation:
                one = one - m_
                concated = concated - np.tile(m_, deep)

            r1, r2 = np.meshgrid(one, concated)
            r = r1 * r2
            if tmp is None:
                tmp = r
            else:
                tmp = tmp + r

        res2 = tmp / (matrix_all_len - deep + 1)

        if (deep == 3) and multiply:
            res_len = len(res2[0])
            r1 = res2[0:res_len]
            r2 = res2[res_len:(2 * res_len)]
            r3 = res2[(2 * res_len):(3 * res_len)]
            w_1_1 = r1
            w_1_2 = r2
            w_1_3 = r3
            w_2_1 = np.rot90(np.fliplr(r2))
            w_2_2 = r1
            w_2_3 = r2
            w_3_1 = np.rot90(np.fliplr(r3))
            w_3_2 = np.rot90(np.fliplr(r2))
            w_3_3 = r1

            w1 = np.concatenate((w_1_1, w_1_2, w_1_3))
            w2 = np.concatenate((w_2_1, w_2_2, w_2_3))
            w3 = np.concatenate((w_3_1, w_3_2, w_3_3))

            res2 = np.concatenate((w1, w2, w3), axis=1)
        
        return res2
    
    def combine_columns(self, row):
        all_data_array = np.array([])
        for item in row:
            all_data_array = np.concatenate((all_data_array, item))
        return all_data_array

    def MyVar(self, a):
        return np.var(np.array(a))

    def getStatistics(self, i_statistics, tw_statistics, i_data, o_data, y_time, d, name, ytext, diff_ytext):
        diff_mathematical_statistics = i_statistics - tw_statistics
        print(f"Root mean square distances: {np.sqrt(np.mean((self.get_flatten(diff_mathematical_statistics)**2)))}")
        i4_start = tt_time.time()
        i_mean_back, i_mean_time = self.interp_matrix_back(i_data, i_statistics, ddof=d.iloc[0, 0])
        i4_end = tt_time.time()
        # self.plot_to_png(f"i {name} ecg {self.ecg_config.getSigName()}", i_mean_time, i_mean_back, xlim=(None, 5.2), xtext="$t, s$", ytext=ytext)
        tw4_start = tt_time.time()
        tw_mean_back = [*self.get_flatten(tw_statistics)] * len(o_data)
        tw_mean_time = self.get_flatten(y_time)
        tw4_end = tt_time.time()
        self.plot_to_png(f"tw {name} {self.ecg_config.getSigName()}", tw_mean_time, tw_mean_back, xlim=(None, 100), xtext="$t, Hour$", ytext=ytext)
        diff_i_mean_back = [*self.get_flatten(diff_mathematical_statistics)] * len(o_data)
        # diff_i_mean_back, diff_i_mean_time = self.interp_matrix_back(i_data, diff_mathematical_statistics, ddof=d.iloc[0, 0])
        # self.plot_to_png(f"diff {name} ecg {self.ecg_config.getSigName()}", tw_mean_time, diff_i_mean_back, xlim=(None, 5.2), xtext="$t, s$", ytext=diff_ytext)
        self.i4 = i4_end - i4_start
        self.tw4 = tw4_end - tw4_start

    def interp_matrix_back(self, i_data, i_mean, ddof = 0):
        df = pd.DataFrame(index=np.arange(len(i_data)), columns=["data_m_1", "data_m_2", "data_m_3"])
        for m in np.arange(0, df.shape[0]):
            for i in np.arange(0, df.shape[1]):
                if m == 0:
                    df.iat[m, i] = i_mean.iat[i]
                    continue
                arr_interp = interp1d(np.arange(df.iat[0, i].size), df.iat[0, i])
                df.iat[m, i] = arr_interp(np.linspace(0, df.iat[0, i].size - 1, i_data.iat[m, i].size))
        f_df = self.get_flatten(df)
        time = (np.arange(0, len(f_df), 1) / self.sampling_rate) + ddof
        return f_df, time

    def get_flatten(self, data):
        all_data_array = np.array([])
        for values in data.values.flatten():
            all_data_array = np.concatenate((all_data_array, values))
        return all_data_array

    def get_Psi(self, d, z, alpha, beta, t__ = False):
        y_2 = []
        y_1 = []
        x_1 = []
        t_ = 0
        for m in np.arange(0, z.shape[0]):
            for i in np.arange(1, z.shape[1]):
                W_i = self.get_W(d.iat[m, i - 1], d.iat[m, i])
                if t__ :
                    t_ = W_i
                x_1.append(W_i)
                y_1.append(((alpha.iat[m, i-1] * W_i) + beta.iat[m, i-1]) - t_)
                y_2.append([alpha.iat[m, i-1]] * len(W_i))
        self.Nu_ = (y_2, x_1)
        return y_2, y_1, x_1


    def getY_1(self, t, d, c, g, t__ = False):
        y_1 = []
        x_1 = []
        t_ = 0
        for m in np.arange(0, t.shape[0]):
            for i in np.arange(1, t.shape[1]):
                W_i = self.get_W(d.iat[m, i - 1], d.iat[m, i])
                if t__ :
                    t_ = W_i
                x_1.append(W_i)
                y_1.append(((c.iat[m, i-1] * W_i) + g.iat[m, i-1]) - t_)

        # for m in np.arange(0, d.shape[0]):
        #     for i in np.arange(1, d.shape[1]):
        #         # W_i = self.get_W(d.iat[m, i - 1], d.iat[m, i])
        #         W_i = np.array([d.iat[m, i - 1], d.iat[m, i]])
        #         x_1.append(W_i)
        #         y_1.append(((c.iat[m, i-1] * W_i) + g.iat[m, i-1]))
        return y_1, x_1

    def getY_(self, t, a, b, t__ = False):
        y_1 = []
        x_1 = []
        t_ = 0
        for m in np.arange(0, t.shape[0]):
            for i in np.arange(1, t.shape[1]):
                W_i = self.get_W(t.iat[m, i - 1], t.iat[m, i])
                if t__ :
                    t_ = W_i
                x_1.append(W_i)
                y_1.append(((a.iat[m, i-1] * W_i) + b.iat[m, i-1]) - t_)
        return y_1, x_1


    def plot_to_png(self, name, x_1, y_1, xlim = None, ylim = None, ytext=[""], xtext="", size=(19, 6), piece = False):
        logger.info(f"Plot {name}.png")
        plt.clf()
        plt.rcParams.update({'font.size': 22})
        f, axis = plt.subplots(1)
        # f.tight_layout()
        f.set_size_inches(size[0], size[1])
        axis.grid(True)
        if piece:
            for x_1__, y_1__ in zip(x_1, y_1):
                axis.plot(x_1__, y_1__, linewidth=3, color = '#1f77b4')
        else:
            axis.plot(x_1, y_1, linewidth=3)
        axis.set_xlabel(xtext, loc = 'right')
        axis.legend(ytext, loc='upper right')
        if not piece:
            axis.axis(xmin = x_1[0], xmax = x_1[-1])
        if xlim is not None:
            axis.axis(xmin = xlim[0], xmax = xlim[1])
        if ylim is not None:
            axis.axis(ymin = ylim[0], ymax = ylim[1])
        f.tight_layout()
        plt.savefig(f'{self.tw_path}/{name}.png', dpi=300)

    def plot_to_2png(self, name, x_1, y_1, x_2, y_2, xlim = None, ylim = None, ytext=[""], xtext="", size=(19, 6), piece = False):
        logger.info(f"Plot {name}.png")
        plt.clf()
        plt.rcParams.update({'font.size': 20})
        f, axis = plt.subplots(1)
        f.set_size_inches(size[0], size[1])
        f.tight_layout()
        axis.grid(True)
        if piece:
            for x_1__, y_1__ in zip(x_1, y_1):
                axis.plot(x_1__, y_1__, linewidth=3, color = '#1f77b4')
        else:
            axis.plot(x_2, y_2, linewidth=3)
            axis.plot(x_1, y_1, linewidth=3)
        axis.set_xlabel(xtext, loc = 'right')
        axis.legend(ytext, loc='upper right')
        if not piece:
            axis.axis(xmin = min(x_1[0], x_2[0]), xmax = max(x_1[-1], x_2[-1]))
        if xlim is not None:
            axis.axis(xmin = xlim[0], xmax = xlim[1])
        if ylim is not None:
            axis.axis(ymin = ylim[0], ymax = ylim[1])
        plt.savefig(f'{self.tw_path}/{name}.png', dpi=300)

    def _3d_plot_to_file(self, plot1, f_time, name, size=(10, 10, 10), ztext="", v = None):
        logger.info(f"Plot {name}.png")
        plt.rcParams.update({'font.size': 13})
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        fig.set_size_inches(size[0], size[1], size[2])
        t1 = f_time[0: len(plot1[0])]
        t2 = f_time[0: len(plot1)]
        t1, t2 = np.meshgrid(t1, t2)
        ax.set_xlabel('$t_1, Hour$')
        ax.set_ylabel('$t_2, Hour$')
        ax.set_zlabel(ztext)
        # Plot the surface.
        if v is None:
            surf = ax.plot_surface(t1, t2, plot1, rstride=5,cstride=5,cmap=cm.coolwarm,linewidth=0)
        else:
            surf = ax.plot_surface(t1, t2, plot1, rstride=5,cstride=5,cmap=cm.coolwarm,linewidth=0, vmin = v[0], vmax = v[1])

        plt.gca().invert_xaxis()
        plt.savefig(f'{self.tw_path}/{name}.png', dpi=600)

    def get_W(self, start, end):
        step = 1.0 / self.sampling_rate
        start_rounded = np.round(start, 6)
        end_rounded = np.round(end - step, 6)
        times = np.arange(start_rounded, end_rounded + step, step)
        if times[-1] > end_rounded:
            times = times[times <= end_rounded]
        return times
    
    def get_main_data(self):
        self.getData(None)
        self.mod_sampling_rate = int(self.sampling_rate * self.ecg_config.getMultiplier())
        self.i1_start = tt_time.time()

        matrix_T_P_size = self.getNewMatrixSize(self.matrix_T_P)
        matrix_P_R_size = self.getNewMatrixSize(self.matrix_P_R)
        matrix_R_T_size = self.getNewMatrixSize(self.matrix_R_T)

        interp_matrix_T_P = []
        interp_matrix_P_R = []
        interp_matrix_R_T = []

        for i in range(len(self.matrix_T_P)):
            arr = np.array(self.matrix_T_P[i])
            arr_interp = interp1d(np.arange(arr.size), arr)
            arr_stretch = arr_interp(np.linspace(0, arr.size - 1, matrix_T_P_size))
            interp_matrix_T_P.append(arr_stretch)

        for i in range(len(self.matrix_P_R)):
            arr = np.array(self.matrix_P_R[i])
            arr_interp = interp1d(np.arange(arr.size), arr)
            arr_stretch = arr_interp(np.linspace(0, arr.size - 1, matrix_P_R_size))
            interp_matrix_P_R.append(arr_stretch)

        for i in range(len(self.matrix_R_T)):
            arr = np.array(self.matrix_R_T[i])
            arr_interp = interp1d(np.arange(arr.size), arr)
            arr_stretch = arr_interp(np.linspace(0, arr.size - 1, matrix_R_T_size))
            interp_matrix_R_T.append(arr_stretch)

        interp_matrix_all = {
            "data_m_1": [*np.array(interp_matrix_P_R)],
            "data_m_2": [*np.array(interp_matrix_R_T)],
            "data_m_3": [*np.array(interp_matrix_T_P)]
        }

        return pd.DataFrame(interp_matrix_all)

    def getNu_(self):
        return self.Nu_

    def getNewMatrixSize(self, matrix):
        n = int(len(matrix[0]) * self.ecg_config.getMultiplier())
        return n





