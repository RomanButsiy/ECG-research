import argparse
import numpy as np
import pandas as pd
import wfdb
from loguru import logger
import scipy.io

def mat_record_to_csv(record_path: str, output_csv: str):
    """
    Читає PhysioNet-запис (rdsamp), будує масив часу і зберігає у CSV.
    Перший стовпець — Time, інші — канали (sig_name).
    """
    logger.info(f"Reading PhysioNet record from: {record_path}")
    data = scipy.io.loadmat(record_path)
    signals = data["signal"]
    fields = [arr.item().strip() for arr in data["siginfo"]["Description"][0]]
    # signals: numpy array форми (n_samples, n_channels)
    fs = int((data["Fs"])[0][0])  # sampling rate
    sig_names = fields  # список імен каналів, довжина = n_channels

    n_samples = signals.shape[0]
    logger.info(f"Number of samples: {n_samples}")
    logger.info(f"Number of channels: {len(sig_names)}")
    logger.info(f"Sampling rate: {fs} Hz")
    logger.info(f"Channel names: {sig_names}")

    # Формуємо колонку часу (в секундах), як 0, 1/fs, 2/fs, ..., (n_samples-1)/fs
    time_vector = np.arange(n_samples) / fs

    # Об'єднуємо time_vector і сигнали у єдиний масив (n_samples, 1 + n_channels)
    data_all = np.column_stack((time_vector, signals))

    # Будуємо DataFrame
    column_names = ['Time'] + sig_names
    df = pd.DataFrame(data_all, columns=column_names)

    # Зберігаємо у CSV без індексу
    output_xlsx = output_csv.replace('.csv', '.xlsx')
    logger.info(f"Writing to Excel: {output_xlsx}")
    df.to_excel(output_xlsx, index=False)
    logger.success("Conversion to .xlsx completed.")

def physionet_record_to_csv(record_path: str, output_csv: str):

    logger.info(f"Reading Mat record from: {record_path}")
    signals, fields = wfdb.rdsamp(record_path)
    # signals: numpy array форми (n_samples, n_channels)
    fs = fields['fs']  # sampling rate
    sig_names = fields['sig_name']  # список імен каналів, довжина = n_channels

    n_samples = signals.shape[0]
    logger.info(f"Number of samples: {n_samples}")
    logger.info(f"Number of channels: {len(sig_names)}")
    logger.info(f"Sampling rate: {fs} Hz")
    logger.info(f"Channel names: {sig_names}")

    # Формуємо колонку часу (в секундах), як 0, 1/fs, 2/fs, ..., (n_samples-1)/fs
    time_vector = np.arange(n_samples) / fs

    # Об'єднуємо time_vector і сигнали у єдиний масив (n_samples, 1 + n_channels)
    data_all = np.column_stack((time_vector, signals))

    # Будуємо DataFrame
    column_names = ['Time'] + sig_names
    df = pd.DataFrame(data_all, columns=column_names)

    # Зберігаємо у CSV без індексу
    logger.info(f"Writing to CSV: {output_csv}")
    df.to_csv(output_csv, index=False)
    logger.success("Conversion completed.")


def main():
    parser = argparse.ArgumentParser(
        description="Конвертація PhysioNet-запису (WFDB) у CSV. "
                    "Перший стовпець — Time (с), далі — дані каналів."
    )
    parser.add_argument(
        "record_path",
        help="Шлях до фізіонет-запису (без розширення). Наприклад: 'mitdb/100'."
    )
    parser.add_argument(
        "-o", "--output",
        default="output.csv",
        help="Ім'я вихідного CSV-файлу (за замовчуванням: output.csv)."
    )
    args = parser.parse_args()

    # physionet_record_to_csv(args.record_path, args.output)
    mat_record_to_csv(args.record_path, args.output)


if __name__ == "__main__":
    main()


# import pandas as pd
# import matplotlib.pyplot as plt

# df = pd.read_csv('m001_1.csv')
# x = df.iloc[:, 0]
# y = df.iloc[:, 1]

# plt.figure()
# plt.plot(x, y)
# plt.xlabel('Час')
# plt.ylabel('Сигнал')
# plt.title('Сигнал за часом')
# plt.show()
