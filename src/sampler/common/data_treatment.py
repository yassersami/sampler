import warnings
from typing import List, Dict

import numpy as np
import pandas as pd

from sampler.common.scalers import MixedMinMaxScaler


class DataTreatment:
    def __init__(
            self,
            features: List[str],
            targets: List[str],
            scaler: MixedMinMaxScaler,
            outliers_filling: Dict,
            variables_ranges: Dict,
            sim_time_cutoff: int,
            interest_region: Dict,
    ):
        self.features = features
        self.targets = targets
        self.scaler = scaler
        self.defaults = outliers_filling["default_values"]
        self.variables_ranges = variables_ranges
        self.sim_time_cutoff = sim_time_cutoff
        self.interest_region = interest_region
        self.scaled_interest_region = scale_interest_region(interest_region, self.scaler)

    def treat_data(self, df_real: pd.DataFrame, scale=True) -> pd.DataFrame:
        """
        Treat and optionally scale data.

        Args:
            df_real (pd.DataFrame): DataFrame containing at least features and targets in the real space (not scaled one).
            scale (bool): Whether to scale the data or not. Defaults to True.

        Returns:
            Tuple[pd.DataFrame, np.ndarray]: Tuple of (treated_data, treated_errors)
        """
        if len(df_real) == 0:
            return df_real
        # Get outliers
        real_data, errors = self.treat_outliers(df=df_real)
        if not scale:
            return real_data, errors
        scaler_cols = self.features + self.targets
        # Scale clean data, contains no nans
        scaled_data = real_data.copy()
        scaled_data[scaler_cols] = self.scaler.transform(real_data[scaler_cols].values)
        # Scale errors, but don't forget to fill nans temporarily
        scaled_errors = errors.copy()
        scaled_errors[scaler_cols] = self.scaler.transform_with_nans(errors[scaler_cols].values)
        return scaled_data, scaled_errors

    def treat_outliers(self, df: pd.DataFrame) -> tuple:
        """
        Treats outliers in the input DataFrame following these steps:
        1. Drop rows with out-of-bounds features
        2. Split rows with time_out or sim_error from the rest
        3. Fill out-of-bounds targets with fixed values

        Parameters:
        - df (pd.DataFrame): Input DataFrame containing simulation data with at least
        features and targets both in the real space (not scaled one)

        Returns:
        tuple: (res, error_res)
            - res: DataFrame with treated data
            - error_res: DataFrame with time_out and sim_error rows
        """
        outliers = self.get_outliers(df=df)
        res = df.copy()

        # Step 1: Drop rows with out-of-bounds features
        if outliers["out_of_bounds_feat"].any():
            warnings.warn(f"There are samples with features out of design space bounds: \n{res[outliers['out_of_bounds_feat']]}")
            res = res[~outliers["out_of_bounds_feat"]]
        
        if len(res) == 0:
            return res, pd.DataFrame()

        # Step 2: Split rows with time_out or sim_error
        error_mask = outliers["time_out"] | outliers["sim_error"]
        error_res = res[error_mask]
        res = res[~error_mask]

        # Step 3: Fill out-of-bounds targets with fixed values
        res = self.fill_outliers_with_fixed_value(
            outliers_mask=outliers["out_of_bounds_tar"],
            df=res
        )

        return res, error_res

    def get_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        # Tolerance to consider a value outside its bounds
        tol = 1e-16
        
        # Initialize DataFrame to store masks
        masks = pd.DataFrame(index=df.index)
        
        # Initialize feature and target masks
        masks['out_of_bounds_feat'] = False
        masks['out_of_bounds_tar'] = False
        
        for key, val in self.variables_ranges.items():
            bounds = [val["bounds"][0] - tol, val["bounds"][1] + tol]
            mask = ~(df[key].between(*bounds))
            
            if key in self.features:
                masks['out_of_bounds_feat'] |= mask
            elif key in self.targets:
                masks['out_of_bounds_tar'] |= mask
        
        # Create masks for time_out and sim_error
        masks['time_out'] = df["sim_time"].round() >= self.sim_time_cutoff
        masks['sim_error'] = df[self.targets].isna().any(axis=1)
        
        return masks

    def fill_outliers_with_fixed_value(self, outliers_mask: np.ndarray, df: pd.DataFrame):
        # Create a DataFrame with default values for all targets
        default_values = pd.DataFrame(
            data=[[self.defaults[t] for t in self.targets]],
            columns=self.targets,
            index=df.index,
        )
        
        # Use the mask to replace values in the original DataFrame
        df.loc[outliers_mask, self.targets] = default_values.loc[outliers_mask]
        
        return df

    def define_interest(self, data: pd.DataFrame):
        aux_df = pd.DataFrame()
        for key, val in self.interest_region.items():
            aux_df[key] = (val[0] < data[key]) & (data[key] < val[1])
        interest_cond = aux_df.all(axis=1)
        data["quality"] = np.where(interest_cond, "interest", "not_interesting")
        return data

    def classify_scaled_interest(self, data: pd.DataFrame):
        aux_df = pd.DataFrame()
        for key, val in self.scaled_interest_region.items():
            aux_df[key] = (val[0] < data[key]) & (data[key] < val[1])
        interest_cond = aux_df.all(axis=1)
        data["quality"] = np.where(interest_cond, "interest", "not_interesting")
        return data

    def define_errors(self, data: pd.DataFrame):
        outliers = self.get_outliers(df=data)
        for key, val in outliers.items():
            data.loc[val, "quality"] = key
        return outliers

    def define_quality_of_data(self, data: pd.DataFrame):
        data = self.define_interest(data)
        errors = self.define_errors(data)
        return data, errors


def scale_interest_region(interest_region: Dict, scaler: MixedMinMaxScaler) -> Dict:
    """ Scale values of the region of interest"""

    lowers = [region[0] for region in interest_region.values()]
    uppers = [region[1] for region in interest_region.values()]
    scaled_bounds = scaler.transform_targets(X_tar=np.array([lowers, uppers]))

    scaled_interest_region = {
        key: list(scaled_bounds[:, i])
        for i, key in enumerate(interest_region.keys())
    }

    for key, scaled_values in scaled_interest_region.items():
        assert all(0 <= val <= 1 for val in scaled_values), (
            f'Error! Region bounds {scaled_values} for key "{key}" not in range [0, 1]!'
            + '\n prep.json must contain absolute bounds! (ie. all data and interest '
            + 'regions values must be INSIDE those bounds)'
        )

    return scaled_interest_region


def initialize_dataset(
    data: pd.DataFrame, features: List[str], targets: List[str], treatment: DataTreatment
) -> pd.DataFrame:
    """
    The data here is already scaled (features and targets)
        1. Create results DataFrame, containing essential columns
        2. Classify initial samples by setting quality column
    """
    # Add column 'quality' with 'interest' if row is inside the interest region
    df_res = data[features+targets].copy(deep=True)
    df_res = treatment.classify_scaled_interest(df_res)
    return df_res
