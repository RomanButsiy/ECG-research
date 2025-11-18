#!/bin/bash

. env/bin/activate

python src/main.py -c 0_H_P002_PPG_S_S2 time-warping
python src/main.py -c 0_H_P003_PPG_S_S2 time-warping
python src/main.py -c 0_H_P004_PPG_S_S2 time-warping
python src/main.py -c 0_H_P005_PPG_S_S2 time-warping
python src/main.py -c 0_H_P006_PPG_S_S2 time-warping
python src/main.py -c 0_H_P010_PPG_S_S2 time-warping
python src/main.py -c 0_H_P011_PPG_S_S2 time-warping
python src/main.py -c 0_H_P013_PPG_S_S2 time-warping
python src/main.py -c 0_H_P015_PPG_S_S2 time-warping
python src/main.py -c 0_H_P016_PPG_S_S2 time-warping
python src/main.py -c 0_H_P017_PPG_S_S2 time-warping
python src/main.py -c 0_H_P019_PPG_S_S2 time-warping

# python src/main.py -c H_P013_M013_1_S2  plot-statistics
# python src/main.py -c H_P013_M013_2_S2  plot-statistics
# python src/main.py -c H_P013_M013_3_S2  plot-statistics
# python src/main.py -c H_P013_M013_4_S2  plot-statistics
# python src/main.py -c H_P013_M013_5_S2  plot-statistics
# python src/main.py -c H_P013_M013_6_S2  plot-statistics
# python src/main.py -c H_P013_M013_7_S2  plot-statistics
# python src/main.py -c H_P013_M013_8_S2  plot-statistics
# python src/main.py -c H_P013_M013_9_S2  plot-statistics
# python src/main.py -c H_P013_M013_10_S2 plot-statistics

# python src/main.py -c H_P001 authentication-test
# python src/main.py -c H_P002 authentication-test
# python src/main.py -c H_P003 authentication-test
# python src/main.py -c H_P005 authentication-test
# python src/main.py -c H_P006 authentication-test
# python src/main.py -c H_P007 authentication-test
# python src/main.py -c H_P008 authentication-test
# python src/main.py -c H_P009 authentication-test
# python src/main.py -c H_P010 authentication-test
# python src/main.py -c H_P011 authentication-test
# python src/main.py -c H_P012 authentication-test
# python src/main.py -c H_P013 authentication-test
# python src/main.py -c H_P014 authentication-test
# python src/main.py -c H_P015 authentication-test
# python src/main.py -c H_P016 authentication-test
# python src/main.py -c H_P017 authentication-test
# python src/main.py -c H_P019 authentication-test
# python src/main.py -c H_P020 authentication-test

# python src/main.py -c 1_P_П1_Екс_1хв_2сигн_AHA_0201++ plot-statistics
# python src/main.py -c 1_P_П3_РитмAF_шк_1_хв_2сигн_N_03++ plot-statistics
# python src/main.py -c 2_Р_П5_2_Ш_екстр_10c_2сигн_М-В_Supr_AR_812++ plot-statistics
# python src/main.py -c 2_P_П1_Екс_1хв_1сигн_ANSI_3b++ plot-statistics
# python src/main.py -c 2_P_П3_РитмAF_шк_1_хв_2сигн_N_07++ plot-statistics
# python src/main.py -c 3_Р_П5_3_Надш_екстр_10c_2сигн_М-В_Supr_AR_820++ plot-statistics
# python src/main.py -c 3_P_П1_Екс_1хв_1сигн_AHSI_3с++ plot-statistics
# python src/main.py -c 3_P_П3_РитмAF_шк_1_хв_2сигн_N_08++ plot-statistics
# python src/main.py -c 4_Р_П_8_Спарена_екстр_BIDM_PPG_Resp_1_хв_5сигн_26++ plot-statistics
# python src/main.py -c 4_P_П1_Екс_1хв_1сигн_ANSI_3d++ plot-statistics
# python src/main.py -c 5_Р_П_8_Част_шл_екстр_BIDM_PPG_Resp_1_хв_5сигн_33++ plot-statistics
# python src/main.py -c 5_P_П1_Екс_1хв_2_сигн_BIDMC_06++ plot-statistics
# python src/main.py -c 6_М_П_4_Морфологія_блокада_PTB_1_хв_12сигн_014++ plot-statistics
# python src/main.py -c 7_М_П_4_Морфологія_блокада_PTB_1_хв_12сигн_015++ plot-statistics

# python src/main.py -c 1_P_П1_Екс_1хв_2сигн_AHA_0201++ plot-fourier-statistics
# python src/main.py -c 1_P_П3_РитмAF_шк_1_хв_2сигн_N_03++ plot-fourier-statistics
# python src/main.py -c 2_Р_П5_2_Ш_екстр_10c_2сигн_М-В_Supr_AR_812++ plot-fourier-statistics
# python src/main.py -c 2_P_П1_Екс_1хв_1сигн_ANSI_3b++ plot-fourier-statistics
# python src/main.py -c 2_P_П3_РитмAF_шк_1_хв_2сигн_N_07++ plot-fourier-statistics
# python src/main.py -c 3_Р_П5_3_Надш_екстр_10c_2сигн_М-В_Supr_AR_820++ plot-fourier-statistics
# python src/main.py -c 3_P_П1_Екс_1хв_1сигн_AHSI_3с++ plot-fourier-statistics
# python src/main.py -c 3_P_П3_РитмAF_шк_1_хв_2сигн_N_08++ plot-fourier-statistics
# python src/main.py -c 4_Р_П_8_Спарена_екстр_BIDM_PPG_Resp_1_хв_5сигн_26++ plot-fourier-statistics
# python src/main.py -c 4_P_П1_Екс_1хв_1сигн_ANSI_3d++ plot-fourier-statistics
# python src/main.py -c 5_Р_П_8_Част_шл_екстр_BIDM_PPG_Resp_1_хв_5сигн_33++ plot-fourier-statistics
# python src/main.py -c 5_P_П1_Екс_1хв_2_сигн_BIDMC_06++ plot-fourier-statistics
# python src/main.py -c 6_М_П_4_Морфологія_блокада_PTB_1_хв_12сигн_014++ plot-fourier-statistics
# python src/main.py -c 7_М_П_4_Морфологія_блокада_PTB_1_хв_12сигн_015++ plot-fourier-statistics

# python src/main.py -c 1_P_П1_Екс_1хв_2сигн_AHA_0201++ show-intervals
# python src/main.py -c 1_P_П3_РитмAF_шк_1_хв_2сигн_N_03++ show-intervals
# python src/main.py -c 2_Р_П5_2_Ш_екстр_10c_2сигн_М-В_Supr_AR_812++ show-intervals
# python src/main.py -c 2_P_П1_Екс_1хв_1сигн_ANSI_3b++ show-intervals
# python src/main.py -c 2_P_П3_РитмAF_шк_1_хв_2сигн_N_07++ show-intervals
# python src/main.py -c 3_Р_П5_3_Надш_екстр_10c_2сигн_М-В_Supr_AR_820++ show-intervals
# python src/main.py -c 3_P_П1_Екс_1хв_1сигн_AHSI_3с++ show-intervals
# python src/main.py -c 3_P_П3_РитмAF_шк_1_хв_2сигн_N_08++ show-intervals
# python src/main.py -c 4_Р_П_8_Спарена_екстр_BIDM_PPG_Resp_1_хв_5сигн_26++ show-intervals
# python src/main.py -c 4_P_П1_Екс_1хв_1сигн_ANSI_3d++ show-intervals
# python src/main.py -c 5_Р_П_8_Част_шл_екстр_BIDM_PPG_Resp_1_хв_5сигн_33++ show-intervals
# python src/main.py -c 5_P_П1_Екс_1хв_2_сигн_BIDMC_06++ show-intervals
# python src/main.py -c 6_М_П_4_Морфологія_блокада_PTB_1_хв_12сигн_014++ show-intervals
# python src/main.py -c 7_М_П_4_Морфологія_блокада_PTB_1_хв_12сигн_015++ show-intervals

# python src/main.py -c 1_P_П1_Екс_1хв_2сигн_AHA_0201++ get-fr-val
# python src/main.py -c 1_P_П3_РитмAF_шк_1_хв_2сигн_N_03++ get-fr-val
# python src/main.py -c 2_Р_П5_2_Ш_екстр_10c_2сигн_М-В_Supr_AR_812++ get-fr-val
# python src/main.py -c 2_P_П1_Екс_1хв_1сигн_ANSI_3b++ get-fr-val
# python src/main.py -c 2_P_П3_РитмAF_шк_1_хв_2сигн_N_07++ get-fr-val
# python src/main.py -c 3_Р_П5_3_Надш_екстр_10c_2сигн_М-В_Supr_AR_820++ get-fr-val
# python src/main.py -c 3_P_П1_Екс_1хв_1сигн_AHSI_3с++ get-fr-val
# python src/main.py -c 3_P_П3_РитмAF_шк_1_хв_2сигн_N_08++ get-fr-val
# python src/main.py -c 4_Р_П_8_Спарена_екстр_BIDM_PPG_Resp_1_хв_5сигн_26++ get-fr-val
# python src/main.py -c 4_P_П1_Екс_1хв_1сигн_ANSI_3d++ get-fr-val
# python src/main.py -c 5_Р_П_8_Част_шл_екстр_BIDM_PPG_Resp_1_хв_5сигн_33++ get-fr-val
# python src/main.py -c 5_P_П1_Екс_1хв_2_сигн_BIDMC_06++ get-fr-val
# python src/main.py -c 6_М_П_4_Морфологія_блокада_PTB_1_хв_12сигн_014++ get-fr-val
# python src/main.py -c 7_М_П_4_Морфологія_блокада_PTB_1_хв_12сигн_015++ get-fr-val

# python src/main.py -c 1_P_П1_Екс_1хв_2сигн_AHA_0201++ plot-fr
# python src/main.py -c 1_P_П3_РитмAF_шк_1_хв_2сигн_N_03++ plot-fr
# python src/main.py -c 2_Р_П5_2_Ш_екстр_10c_2сигн_М-В_Supr_AR_812++ plot-fr
# python src/main.py -c 2_P_П1_Екс_1хв_1сигн_ANSI_3b++ plot-fr
# python src/main.py -c 2_P_П3_РитмAF_шк_1_хв_2сигн_N_07++ plot-fr
# python src/main.py -c 3_Р_П5_3_Надш_екстр_10c_2сигн_М-В_Supr_AR_820++ plot-fr
# python src/main.py -c 3_P_П1_Екс_1хв_1сигн_AHSI_3с++ plot-fr
# python src/main.py -c 3_P_П3_РитмAF_шк_1_хв_2сигн_N_08++ plot-fr
# python src/main.py -c 4_Р_П_8_Спарена_екстр_BIDM_PPG_Resp_1_хв_5сигн_26++ plot-fr
# python src/main.py -c 4_P_П1_Екс_1хв_1сигн_ANSI_3d++ plot-fr
# python src/main.py -c 5_Р_П_8_Част_шл_екстр_BIDM_PPG_Resp_1_хв_5сигн_33++ plot-fr
# python src/main.py -c 5_P_П1_Екс_1хв_2_сигн_BIDMC_06++ plot-fr
# python src/main.py -c 6_М_П_4_Морфологія_блокада_PTB_1_хв_12сигн_014++ plot-fr
# python src/main.py -c 7_М_П_4_Морфологія_блокада_PTB_1_хв_12сигн_015++ plot-fr


# python src/main.py -c LM_4_1_М_П_4_Морфологія_блокада_BID_1_хв_2сигн_01 plot-statistics
# python src/main.py -c LM_7_1_Н_7_Норма_1_хв_BIDM_PPG_Resp_07 plot-statistics
# python src/main.py -c LM_7_2_Н_7_Норма_1_хв_BIDM_PPG_Resp_09 plot-statistics
# python src/main.py -c LM_7_3_Н_7_Норма_1_хв_BIDM_PPG_Resp_34 plot-statistics
# python src/main.py -c LM_8_1_Р_П_8_Част_надшл_екстр_Ch_TSB_2010_1_хв_7сигн_06 plot-statistics
# python src/main.py -c LM_8_2_Р_П_8_Част_надшл_екстр_Ch_TSB_2010_1_хв_7сигн_10 plot-statistics

# python src/main.py -c LM_4_1_М_П_4_Морфологія_блокада_BID_1_хв_2сигн_01 plot-fourier-statistics
# python src/main.py -c LM_7_1_Н_7_Норма_1_хв_BIDM_PPG_Resp_07 plot-fourier-statistics
# python src/main.py -c LM_7_2_Н_7_Норма_1_хв_BIDM_PPG_Resp_09 plot-fourier-statistics
# python src/main.py -c LM_7_3_Н_7_Норма_1_хв_BIDM_PPG_Resp_34 plot-fourier-statistics
# python src/main.py -c LM_8_1_Р_П_8_Част_надшл_екстр_Ch_TSB_2010_1_хв_7сигн_06 plot-fourier-statistics
# python src/main.py -c LM_8_2_Р_П_8_Част_надшл_екстр_Ch_TSB_2010_1_хв_7сигн_10 plot-fourier-statistics

# python src/physionet_to_csv.py /home/MyFiles2/ECG/cebsdb_2minutes/mat/Person_003/m003_1.mat -o m003_1.csv
# python src/physionet_to_csv.py /home/MyFiles2/ECG/cebsdb_2minutes/mat/Person_003/m003_2.mat -o m003_2.csv
# python src/physionet_to_csv.py /home/MyFiles2/ECG/cebsdb_2minutes/mat/Person_003/m003_3.mat -o m003_3.csv
# python src/physionet_to_csv.py /home/MyFiles2/ECG/cebsdb_2minutes/mat/Person_003/m003_4.mat -o m003_4.csv
# python src/physionet_to_csv.py /home/MyFiles2/ECG/cebsdb_2minutes/mat/Person_001/m001_5.mat -o m001_5.csv
# python src/physionet_to_csv.py /home/MyFiles2/ECG/cebsdb_2minutes/mat/Person_001/m001_6.mat -o m001_6.csv
# python src/physionet_to_csv.py /home/MyFiles2/ECG/cebsdb_2minutes/mat/Person_001/m001_7.mat -o m001_7.csv
# python src/physionet_to_csv.py /home/MyFiles2/ECG/cebsdb_2minutes/mat/Person_001/m001_8.mat -o m001_8.csv
# python src/physionet_to_csv.py /home/MyFiles2/ECG/cebsdb_2minutes/mat/Person_001/m001_9.mat -o m001_9.csv
# python src/physionet_to_csv.py /home/MyFiles2/ECG/cebsdb_2minutes/mat/Person_001/m001_10.mat -o m001_10.csv