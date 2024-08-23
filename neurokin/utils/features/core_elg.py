from abc import ABC, abstractmethod
from typing import List, Union

from typeguard import typechecked


class FeatureExtraction(ABC):

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        if not isinstance(cls.extraction_target, str):
            raise TypeError(f"Extraction target type not valid, expected str got {type(cls.extraction_target)}")
        return instance

    @abstractmethod
    def compute_feature(self):
        pass

    def run_feat_extraction(self):
        self.compute_feature()
        return


class MyFeat(FeatureExtraction):
    extraction_target = "markers"

    @typechecked
    def __init__(self, param_a: List[Union[int, str]], param_b: List[int]):
        self.param_a = param_a
        self.param_b = param_b

    def compute_feature(self):
        print("ehi look im doing cool stuff")
        return


class MyFeat2(FeatureExtraction):
    extraction_target = "joints"

    def __init__(self, param_a: ABC, param_b: List[int]):
        self.param_a = param_a
        self.param_b = param_b

    def compute_feature(self):
        pass


if __name__ == "__main__":
    feat = MyFeat(param_a=[1, "str"], param_b=[1, 2, 3])
    feat2 = MyFeat2(param_a=3.0, param_b=[1, 2, 3])
    # feat2.run_feat_extraction()
    feat.run_feat_extraction()
    print(feat.extraction_target)
    # feat.extraction_target = "joints"
    print(feat2.extraction_target)
    # print(feat.extraction_target)
