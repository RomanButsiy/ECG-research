from loguru import logger
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope  # за бажанням

class YLTest():

    def __init__(self, ecg_config):
        self.ecg_config = ecg_config
        logger.debug("YL Init")

        L = int(self.ecg_config.getMultiplier())
        # f_read = f"Zmodelov_Rozkidi_T({L})_T{L}_c70.txt"
        # t_read = f"dla perevirki/Rozkidi_realiz_cicli-ms_peredis_sign__1 (c6_T{L}).txt"
        # t_read = f"dla perevirki/Rozkidi_realiz_cicli-ms_peredis_(TP)Signal_(c7_T{L}).txt"
        # f_read = "navchati/Zmodelov_Rozkidi_T(2690)_Rozk_c100_(T2689).txt"
        # t_read = "testiti/_Rozk_c7_(T2689).txt"
        f_read = "navchati/Zmodelov_Rozkidi_T(1821)rozk_+ 100c_(T1820).txt"
        t_read = "testiti/_+___c8_(T1820).txt"

        data = pd.read_csv(f'{self.ecg_config.getFileName()}/{f_read}',
                           header=None, sep=r'\s+').iloc[:, -1].to_numpy()
        t_data = pd.read_csv(f'{self.ecg_config.getFileName()}/{t_read}',
                             header=None, sep=r'\s+').iloc[:, -1].to_numpy()

        # -> розрізаємо на цикли
        n_full = len(data) // L
        X_train = data[: n_full * L].reshape(n_full, L)

        t_n_full = len(t_data) // L
        X_test = t_data[: t_n_full * L].reshape(t_n_full, L)

        # Загальний препроцес: StandardScaler + PCA (стискаємо до 95% дисперсії)
        preproc = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("pca", PCA(n_components=0.95, svd_solver="full"))
        ])
        X_train_p = preproc.fit_transform(X_train)
        X_test_p  = preproc.transform(X_test)

        # logger.info(f"PCA kept {preproc['pca'].n_components_} comps "
        #             f"out of {X_train.shape[1]} features")
        
        # X_train_p = X_train
        # X_test_p  = X_test

        # Моделі
        models = {
            "OneClassSVM_RBF": OneClassSVM(kernel="rbf", gamma="scale", nu=0.05),
            "IsolationForest": IsolationForest(
                n_estimators=300, contamination=0.05, random_state=42
            ),
            "LOF_novelty": LocalOutlierFactor(
                n_neighbors=35, novelty=True, contamination=0.05
            ),
            "EllipticEnvelope": EllipticEnvelope(contamination=0.05, random_state=42),
        }

        # Фіт + калібровка порогу на train-скорах (q=5%)
        self.fitted = {}
        self.thresholds = {}
        q = 0.05

        for name, clf in models.items():
            logger.info(f"Train start: {name}")
            clf.fit(X_train_p)
            logger.info(f"Train end:   {name}")

            if hasattr(clf, "decision_function"):
                s_train = clf.decision_function(X_train_p)
            elif hasattr(clf, "score_samples"):
                s_train = clf.score_samples(X_train_p)
            else:
                s_train = clf.predict(X_train_p).astype(float)

            thr = np.quantile(s_train, q)
            self.fitted[name] = clf
            self.thresholds[name] = thr
            logger.info(f"{name}: train score Q{int(q*100)}={thr:.5f}")

        # Інференс на тесті
        results = {}
        for name, clf in self.fitted.items():
            y_pred, s_test = self.predict_with_scores(clf, X_test_p)
            thr = self.thresholds[name]
            y_cal = np.where(s_test >= thr, +1, -1)

            # одна підсумкова оцінка (0..1), частка inliers ---
            final_score = float((y_cal == 1).mean())

            results[name] = {
                "y_pred_raw": y_pred,   # мітки від моделі
                "y_pred_thr": y_cal,    # мітки за нашим порогом
                "score":      s_test,   # безпосередні скори
                "thr":        float(thr),
                "final":      final_score,  # <-- ОДНА ОЦІНКА ДЛЯ МОДЕЛІ
            }

        # опційно коротке зведення тільки фінальних оцінок:
        summary = {k: v["final"] for k, v in results.items()}
        print(results)
        print("Final scores:", summary)

    def predict_with_scores(self, clf, X):
        y_pred = clf.predict(X)
        if hasattr(clf, "decision_function"):
            score = clf.decision_function(X)
        elif hasattr(clf, "score_samples"):
            score = clf.score_samples(X)
        else:
            score = (y_pred == 1).astype(float) - 1.0 * (y_pred == -1)
        return y_pred, score
