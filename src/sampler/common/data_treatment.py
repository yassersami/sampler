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

    def select_in_bounds_feat(self, df_real: pd.DataFrame) -> pd.DataFrame:
        """
        Filters the input DataFrame to retain only rows with in-bound features. 
        This function is intended for use when importing new data, such as from
        a CSV file, to ensure that only valid data is processed.
        """
        if df_real.empty:
            return df_real

        # Get masks of all types of outliers
        masks = self.get_outliers_masks(df_real)

        # Keep only rows with in-bound features
        return df_real[~masks["out_of_bounds_feat"]]


    def treat_real_data(self, df_real: pd.DataFrame) -> pd.DataFrame:
        """
        Processes the input DataFrame by filling outlier samples with NaN target
        values and scaling the data. This approach ensures consistency in
        outlier removal by always proceeding to the exclusion of NaNs, thereby
        providing clean data for regression. This NaN exclusion to get 'clean
        data' is utilized in the following contexts:
            - `FigureOfMerit.update`: Extracts `clean_regr_data`.
            - `OutlierExcluder.update_outliers_set`: Extracts `ignored_rows`.
            - Main sampling node: Updates `n_new_inliers`.

        Note: This function does not handle out-of-bounds features, as it is
        assumed that such cases are impossible within the pipeline. The
        `FigureOfMerit.optimize` function is solely responsible for selecting
        new samples and ensures that they remain within the defined design
        space.

        Parameters: - df_real (pd.DataFrame): A DataFrame containing features
        and targets in real (unscaled) space.
        """

        if len(df_real) == 0:
            return df_real
        
        # Get masks of all types of outliers
        masks = self.get_outliers_masks(df_real)
        
        # Check for out-of-bounds features and raise an error if detected
        features_mask = masks["out_of_bounds_feat"]
        if features_mask.any():
            df_out_of_bounds_features = df_real.loc[
                features_mask,
                self.features + self.targets
            ]
            raise ValueError(
                "Out-of-bounds features detected!\n"
                "Affected rows feature and target values:\n"
                f"{df_out_of_bounds_features}\n"
                "Error: `treat_real_data` should never receive samples with "
                "out-of-bounds features. If you are reading CSV data as input, "
                "outside of the pipeline, try cleaning it first with "
                "`select_in_bounds_feat`."
            )
            
        # Fill outliers target values with NaNs
        real_data = self.fill_outliers_with_nans(df_real, masks)
        
        # Scale data
        scaler_cols = self.features + self.targets
        scaled_data = real_data.copy()
        scaled_data[scaler_cols] = self.scaler.transform_with_nans(
            real_data[scaler_cols].values
        )
        return scaled_data

    def fill_outliers_with_nans(self, df_real: pd.DataFrame, masks: Dict) -> pd.DataFrame:
        """
        Treats outliers in the input DataFrame by replacing non-feature values
        with NaNs for non-clean data:
        1. Identify bad rows with out-of-bounds targets, time_out, or sim_error.
        3. Replace target values with NaNs for all bad rows.

        Parameters:
        - df_real (pd.DataFrame): DataFrame containing at least features and
        targets in the real space (not scaled one).

        Returns:
        - pd.DataFrame: DataFrame with the same index as the input, where 
        non-clean data has non-feature values replaced with NaNs.
        """
        df_real_out = df_real.copy()

        # Identify all rows with outliers
        bad_rows_mask = (
            masks["time_out"] |
            masks["sim_error"] |
            masks["out_of_bounds_tar"]
        )

        # Set target columns to NaN for bad rows
        df_real_out.loc[bad_rows_mask, self.targets] = np.nan

        return df_real_out

    def get_outliers_masks(self, df_real: pd.DataFrame) -> pd.DataFrame:
        """ Get outlier classes on real df"""
        # Tolerance to consider a value outside its bounds
        tol = 1e-16
        
        # Initialize DataFrame to store masks
        masks = pd.DataFrame(index=df_real.index)
        
        # Initialize feature and target masks
        masks['out_of_bounds_feat'] = False
        masks['out_of_bounds_tar'] = False
        
        for key, val in self.variables_ranges.items():
            bounds = [val["bounds"][0] - tol, val["bounds"][1] + tol]
            mask = ~(df_real[key].between(*bounds))
            
            if key in self.features:
                masks['out_of_bounds_feat'] |= mask
            elif key in self.targets:
                masks['out_of_bounds_tar'] |= mask
        
        # Create masks for time_out and sim_error
        masks['time_out'] = df_real["sim_time"].round() >= self.sim_time_cutoff
        masks['sim_error'] = df_real[self.targets].isna().any(axis=1)
        
        return masks

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
        outliers_masks = self.get_outliers_masks(real_data)

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
