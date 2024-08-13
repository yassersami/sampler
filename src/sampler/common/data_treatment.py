import warnings
from typing import List, Dict, Tuple

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

        Parameters:
        - df_real (pd.DataFrame): DataFrame containing at least features and
        targets in the real space (not scaled one).
        - scale (bool): Whether to scale the data or not. Defaults to True.

        Returns:
            pd.DataFrame: treated_data
        """
        if len(df_real) == 0:
            return df_real
        # Get outliers
        real_data = self.fill_outliers_with_nans(df_real)
        if not scale:
            return real_data
        scaler_cols = self.features + self.targets
        scaled_data = real_data.copy()
        scaled_data[scaler_cols] = self.scaler.transform_with_nans(
            real_data[scaler_cols].values
        )
        return scaled_data

    def fill_outliers_with_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Treats outliers in the input DataFrame by replacing non-feature values
        with NaNs for non-clean data:
        1. Identify rows with out-of-bounds features, time_out, or sim_error.
        2. Replace non-feature values with NaNs for these rows.

        Parameters:
        - df (pd.DataFrame): Input DataFrame containing simulation data with at 
        least features and targets both in the real space (not scaled one)

        Returns:
        - pd.DataFrame: DataFrame with the same index as the input, where 
        non-clean data has non-feature values replaced with NaNs.
        """
        outliers = self.get_outliers_masks(df=df)
        df_out = df.copy()

        # Identify all rows with outliers
        bad_rows_mask = (
            outliers["out_of_bounds_feat"] |
            outliers["time_out"] |
            outliers["sim_error"] |
            outliers["out_of_bounds_tar"]
        )

        # Set non-feature columns to NaN for bad rows
        # non_feature_columns = df_out.columns.difference(self.features)
        df_out.loc[bad_rows_mask, self.targets] = np.nan

        return df_out

    def get_outliers_masks(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Get outlier classes on real df"""
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

    def split_inliers_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        outliers_masks = self.get_outliers_masks(df=df)
        res = df.copy()

        # Step 1: Drop rows with out-of-bounds features
        if outliers_masks["out_of_bounds_feat"].any():
            warnings.warn(f"There are samples with features out of design space bounds: \n{res[outliers_masks['out_of_bounds_feat']]}")
            res = res[~outliers_masks["out_of_bounds_feat"]]
        
        if len(res) == 0:
            return res, pd.DataFrame()

        # Step 2: Split rows with time_out or sim_error
        error_mask = outliers_masks["time_out"] | outliers_masks["sim_error"]
        error_res = res[error_mask]
        res = res[~error_mask]

        # Step 3: Fill out-of-bounds targets with fixed values
        # TODO yasser: why fill outliers_masks with minimums instead of removing.
        res = self.fill_outliers_with_fixed_value(
            outliers_mask=outliers_masks["out_of_bounds_tar"],
            df=res
        )

        return res, error_res

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

    def classify_quality_interest(
        self, data: pd.DataFrame, data_is_scaled: bool
    ) -> pd.DataFrame:
        """
        Classifies data based on interest or scaled interest regions.

        Returns:
        - The DataFrame with an updated 'quality' column.
        """
        # Choose the appropriate interest region based on whether the data is scaled
        interest_region = self.scaled_interest_region if data_is_scaled else self.interest_region

        # Compute interest condition
        interest_cond = np.all(
            [
                (data[key] > val[0]) & (data[key] < val[1])
                for key, val in interest_region.items()
            ],
            axis=0
        )

        # Set 'quality' column based on interest conditions
        data["quality"] = np.where(interest_cond, "interest", "not_interesting")

        return data

    def classify_quality_error(
        self, data: pd.DataFrame, data_is_scaled: bool
        ) -> pd.DataFrame:
        """
        Classifies data based on outlier masks, with an option to scale the data.

        Returns:
        - The DataFrame with an updated 'quality' column based on outlier
        classification.
        """
        # Create a copy of the data for scaling if necessary
        if data_is_scaled:
            scaler_cols = self.features + self.targets
            # Ensure real_data is initialized before use
            real_data = data.copy()
            # Scale only the specified columns
            real_data[scaler_cols] = self.scaler.transform(data[scaler_cols].values)
        else:
            real_data = data

        # Get outlier masks
        outliers_masks = self.get_outliers_masks(df=real_data)

        # Update 'quality' column based on outlier masks
        for key, mask in outliers_masks.items():
            data.loc[mask, "quality"] = key

        return data

    def define_quality_of_data(self, data: pd.DataFrame, specify_errors=False):
        data = self.classify_quality_interest(data, data_is_scaled=False)
        if specify_errors:
            data = self.classify_quality_error(data, data_is_scaled=False)
        return data


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


def initialize_dataset(data: pd.DataFrame, treatment: DataTreatment) -> pd.DataFrame:
    """
    The data here is already scaled (features and targets)
        1. Create results DataFrame, containing essential columns
        2. Classify initial samples by setting quality column
    """
    # Add column 'quality' with 'interest' if row is inside the interest region
    df_res = data[treatment.features + treatment.targets].copy(deep=True)
    df_res = treatment.classify_quality_interest(df_res, data_is_scaled=True)
    return df_res
