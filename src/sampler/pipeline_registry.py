"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

import sampler.pipelines.prep.pipeline as prep_input
import sampler.pipelines.irbs.pipeline as irbs_sampler


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    pipelines["__default__"] = prep_input.create_pipeline() + irbs_sampler.create_pipeline()
    return pipelines
