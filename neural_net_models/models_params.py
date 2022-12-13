from typing import TypeAlias  # "from typing_extensions" in Python 3.9 and earlier
from dataclasses import dataclass, field
from inspect import stack

from typing import List, Type

from enum import Enum, unique


class ModelsNames(Enum):
    MLP = 1
    CNN_BI_LSTM = 2
    CNN_BI_GRU = 3

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.value == other.value

        return False


@dataclass
class CNNDataInputParams:
    use_gru_instead_of_lstm: bool


@dataclass
class DataInputParams:
    number_of_classes: int = 0
    input_shape: tuple = (0, 0, 0)
    input_features: int = 0
    batch_size: int = 0

    evaluation_function: str = "accuracy"
    USE_AUTOMATIC_MIXED_PRECISION: bool = False


tp: Type[int] = int


# tp: Type[TrialParams] = TrialParams
@dataclass
class MlpTrialParams:
    number_layers: int = 0

    _n_units_l = int
    _dropout_l = int

    # number_layers is the length of the list
    # use stack ops
    stack_layers: List[tuple[_n_units_l, _dropout_l]] = field(
        default_factory=lambda: []
    )


@dataclass
class CnnBiRnnTrialParams:
    number_layers: int = 0

    _n_units_l = int
    _no_strides_l = int

    # number_layers is the length of the list
    # use stack ops
    stack_layers: List[tuple[_n_units_l, _no_strides_l]] = field(
        default_factory=lambda: []
    )

    rnn_drop_procentages: float = 0.0


# ddd=DataInputParams(evaluation_function='f1_score', no_trials=2, USE_AUTOMATIC_MIXED_PRECISION=False)

_alias: Type[DataInputParams] = DataInputParams
_alias = DataInputParams
