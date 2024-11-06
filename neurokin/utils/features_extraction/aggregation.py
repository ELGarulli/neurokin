import pandas as pd
from typeguard import typechecked

from neurokin.utils.features_extraction.core import FeatureExtraction


class Aggregation(FeatureExtraction):
    extraction_target = "markers"

    @typechecked
    def compute_feature(self, df: pd.DataFrame, aggregation_method: str = "mean", window_size: int = 1, step: int = 1):
        if aggregation_method == "mean":
            aggregated_df = df.rolling(window=window_size, step=step).mean()
        elif aggregation_method == "sum":
            aggregated_df = df.rolling(window=window_size, step=step).sum()
        else:
            raise NotImplementedError(f"The aggregation method {aggregation_method} is not yet supported, "
                                      f"please choose between mean and sum")
        aggregated_df = aggregated_df.add_prefix(f"{aggregation_method}_")
        return aggregated_df
