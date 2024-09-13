from typing import List, Dict, Tuple, Union
import numpy as np
import pandas as pd

from .data_treatment import DataTreatment
from .simulator_proxy import FastSimulator
from .simulator_wrapper import run_simulation


class SimulationProcessor:
    def __init__(
        self,
        features:List[str],
        targets: List[str],
        additional_values: List[str],
        treatment: DataTreatment,
        simulator_config: Dict,
        map_dir: str,
        n_proc: int = 1
    ):
        self.features = features
        self.targets = targets
        self.additional_values = additional_values
        self.treatment = treatment
        self.use_simulator = simulator_config['use']
        self.max_simu_time = simulator_config['max_simu_time']
        self.map_dir = map_dir
        self.n_proc = n_proc
        if not self.use_simulator:
            self.proxy = FastSimulator(
                features=self.features,
                targets=self.targets,
                additional_values=self.additional_values,
                interest_region=self.treatment.scaled_interest_region,
            )

    def _prepare_real_input(self, X: np.ndarray, is_real_X: bool) -> np.ndarray:
        if is_real_X:
            return X
        XY = np.column_stack((X, np.zeros((X.shape[0], len(self.targets)))))
        return self.treatment.scaler.inverse_transform(XY)[:, :len(self.features)]

    def process_data(
        self, X: np.ndarray, is_real_X: bool, index: int, treat_output=True
    ) -> pd.DataFrame:
        """
        Process simulation data, either from real or scaled input features.

        Args:
            X (np.ndarray): Input features array.
            is_real_X (bool): If True, X contains real (unscaled) values. If False, X contains scaled values.
            index (int): Index under which to store comming simulation in self.map_dir.

        Returns:
            pd.DataFrame: Processed data either real or treated.
        """
        print(f"{self.__class__.__name__} -> [index: {index}] Running {self.n_proc} simulations...")

        X_real = self._prepare_real_input(X, is_real_X)
        df_X_real = pd.DataFrame(X_real, columns=self.features)

        if self.use_simulator:
            df_results = run_simulation(
                df_X=df_X_real, index=index, n_proc=self.n_proc,
                max_simu_time=self.max_simu_time, map_dir=self.map_dir
            )
        else:
            df_results = self.proxy.run_fast_simulation(X_real, self.treatment.scaler)

        # Concatenate inputs with simulation results
        df_results = pd.concat([df_X_real, df_results], axis=1)

        if not treat_output:
            # Return data in real scale as returned by simulator
            return df_results

        # Scale and clean data
        scaled_data = self.treatment.treat_real_data(df_real=df_results)

        # Keep specific columnms
        available_values = [col for col in self.additional_values if col in df_results]
        scaled_data = scaled_data[self.features + self.targets + available_values]
        return scaled_data

    def adapt_targets(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.use_simulator:
            return data

        # If using fake simulator for rapid testing, change target values
        scaled_data = self.process_data(
            data[self.features].values, is_real_X=False, index=0, treat_output=True
        )
        data[self.targets] = scaled_data[self.targets].values

        # Add some spicy data to check how outlier and interest samples are handled
        data = self.proxy.append_spicy_data(data, self.max_simu_time)

        return data
