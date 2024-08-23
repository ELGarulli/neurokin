import pandas as pd
from typeguard import typechecked
from neurokin.utils.features.core_elg import FeatureExtraction


class Aggregation(FeatureExtraction):
    extraction_target = "markers"

    #def __init__(self):
    #    self.df = df
    #    self.aggregation_method = aggregation_method
    #    self.window_size = window_size

    @typechecked
    def compute_feature(self, df: pd.DataFrame, aggregation_method: str = "mean", window_size: int = 1):
        if aggregation_method == "mean":
            aggregated_df = df.rolling(window=window_size).mean()
        elif aggregation_method == "sum":
            aggregated_df = df.rolling(window=window_size).sum()
        else:
            raise NotImplementedError(f"The aggregation method {aggregation_method} is not yet supported, "
                                      f"please choose between mean and sum")
        aggregated_df.add_prefix(aggregation_method)
        return aggregated_df
