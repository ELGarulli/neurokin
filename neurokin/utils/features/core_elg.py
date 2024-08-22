from abc import ABC, abstractmethod
from typing import List, get_type_hints, get_args, get_origin, final


class FeatureExtraction(ABC):

    def __init__(self, extraction_target_value):
        self.extraction_target = extraction_target_value

    @property
    def extraction_target(self):
        return self._extraction_target

    @extraction_target.setter
    def extraction_target(self, extraction_target_value):
        if isinstance(extraction_target_value, str):
            self._extraction_target = extraction_target_value
        else:
            raise TypeError(f"Extraction target type not valid, expected str got {type(extraction_target_value)}")

    def check_expected_input_types(self):
        type_hints = get_type_hints(self.__init__)
        param_values = self.__dict__

        for key, expected_type in type_hints.items():
            origin = get_origin(expected_type)
            args = get_args(expected_type)
            actual_value = param_values[key]

            if origin:
                if not isinstance(actual_value, origin):
                    raise TypeError(f"Parameter {key} is of incorrect type. "
                                    f"Expected {origin} got {type(actual_value)}")
                if args:
                    for i, item in enumerate(actual_value):
                        if not isinstance(item, args[0]):
                            raise TypeError(f"Argument #{i} of parameter {key} is of incorrect type. "
                                            f"Expected {args[0]} got {type(item)}")
            else:
                if not isinstance(actual_value, expected_type):
                    raise TypeError(f"Parameter {key} is of incorrect type. "
                                    f"Expected {expected_type} got {type(actual_value)}")


    @abstractmethod
    def compute_feature(self):
        pass

    def run_feat_extraction(self):
        self.check_expected_input_types()
        self.compute_feature()
        return



class MyFeat(FeatureExtraction):

    def __init__(self, param_a: float, param_b: List[int]):
        super().__init__(extraction_target_value="markers")
        self.param_a = param_a
        self.param_b = param_b

    def compute_feature(self):
        print("ehi look im doing cool stuff")
        return




class MyFeat2(FeatureExtraction):

    def __init__(self, param_a: float, param_b: List[int]):
        super().__init__(extraction_target_value="joints")
        self.param_a = param_a
        self.param_b = param_b

    def compute_feature(self):
        pass


if __name__ == "__main__":
    feat = MyFeat(param_a=3.0, param_b=[1, 2, 3])
    feat2 = MyFeat2(param_a=3.0, param_b=[1, 2, 3])
    feat2.run_feat_extraction()
    feat.run_feat_extraction()
    print(feat.extraction_target)
    feat.extraction_target = "joints"
    print(feat2.extraction_target)
    print(feat.extraction_target)
