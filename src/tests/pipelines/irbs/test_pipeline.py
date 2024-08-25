"""
This is a boilerplate test file for pipeline 'irbs'
generated using Kedro 0.18.5.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""

import os
import pytest
from kedro.framework.startup import bootstrap_project
from kedro.framework.session import KedroSession

import numpy as np
from scipy.stats.qmc import LatinHypercube

from sampler.fom.base import FOM

# Set project root path
current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(current_dir, "..", "..", "..", ".."))
print(f"Kedro project root: {ROOT}")

@pytest.fixture
def kedro_session():
    """
    Launch with:
    pytest -s
    src/tests/pipelines/irbs/test_pipeline.py::test_fom_class_initialization
    """
    # Bootstrap the Kedro project
    metadata = bootstrap_project(ROOT)
    
    # Create and return a KedroSession
    with KedroSession.create(
        metadata.package_name, metadata.project_path, env="env_test"
    ) as session:
        yield session

def test_fom_class_initialization(kedro_session):
    # Load the context from the session
    context = kedro_session.load_context()
    catalog = context.catalog
    
    # Load parameters
    params = catalog.load('parameters')
    
    # Extract required parameters
    features = params["features"]
    targets = params["targets"]
    interest_region = params["interest_region"]
    terms_config = params["irbs_fom_terms"]
    
    # Initialize FOM class
    fom_instance = FOM(
        features=features,
        targets=targets,
        interest_region=interest_region,
        terms_config=terms_config
    )
    
    # Active terms assertion
    assert len(fom_instance.terms) == sum([v['apply'] for v in terms_config.values()]), (
        "Number of terms mismatch"
    )
    n_active_terms = len(fom_instance.terms)
    
    # Set fake simulator
    def multi_dim_x_sin_x(X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return np.mean(0.5 * (X * np.sin(10 * np.pi * X**2) + 1), axis=1)
    
    # Set train and test datasets
    n_train, n_test, p = 20, 5, 3
    sampler = LatinHypercube(p)
    X_train = sampler.random(n_train)
    y_train = multi_dim_x_sin_x(X_train)
    X_test = sampler.random(n_test)
    
    # Set an outlier case
    X_test[1] = X_train[1]
    y_train[1] = np.nan

    print(f"X_train {X_train.shape}: \n{X_train[:5]}")
    print(f"y_train {y_train.shape}: \n{y_train[:5]}")

    # Fit the FOM
    fom_instance.fit(X_train, y_train)
    
    scores = fom_instance.predict_score(X_test)
    scores_df = fom_instance.predict_scores_df(X_test)
    print(f"Predicted Scores Shape: {scores.shape}")
    print(f"Scores DataFrame: \n{scores_df.head()}")

    # Test shapes
    assert scores.shape == (n_test,), (
        f"Expected scores shape ({n_test},), but got {scores.shape}"
    )
    assert scores_df.shape == (n_test, n_active_terms), (
        f"Expected scores_df shape ({n_test}, {n_active_terms}), "
        f"but got {scores_df.shape}"
    )
    # Outlier proximity detection
    assert scores_df.loc[1, "outlier_proximity"] == 0, (
        "Artificially set outlier at position 1 was not detected."
    )

    fom_term_params = fom_instance.get_parameters()
    print(f"All terms parameters: \n{fom_term_params}")
