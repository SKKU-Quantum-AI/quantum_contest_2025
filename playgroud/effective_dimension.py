import logging
import time
from typing import Any, Union, List, Tuple

import numpy as np
from scipy.special import logsumexp

from ..utils import algorithm_globals
from ..exceptions import QiskitMachineLearningError
from .estimator_qnn import EstimatorQNN
from .neural_network import NeuralNetwork

logger = logging.getLogger(__name__)

class EffectiveDimension:
    def __init__(
        self,
        qnn: NeuralNetwork,
        weight_samples: Union[np.ndarray, int] = 1,
        input_samples: Union[np.ndarray, int] = 1,
    ) -> None:

        self._weight_samples = np.asarray([0.25])
        self._input_samples = np.asarray([0.5])
        self._num_weight_samples = len(self._weight_samples)
        self._num_input_samples = len(self._input_samples)
        self._model = qnn

        self.weight_samples = weight_samples
        self.input_samples = input_samples

    # 그냥 weight 샘플들 반환
    @property
    def weight_samples(self) -> np.ndarray:
        return self._weight_samples

    # weight 샘플 setter임
    # 숫자로 들어오면 랜덤하게 uniform에 따라서 뽑음
    # Union[np.ndarray, int] 라고 하면, type hint 문법인데, np.ndarray 이거나, int 일거라는 거임 
    @weight_samples.setter
    def weight_samples(self, weight_samples: Union[np.ndarray, int]) -> None:
        if isinstance(weight_samples, int):
            self._weight_samples = algorithm_globals.random.uniform(
                0, 1, size=(weight_samples, self._model.num_weights)
            )
            # 0은 평균, 1은 표준편차
            # num_weights는 내가 만든 회로에 세타, 즉 웨이트 전체 개수
            # weight_samples에 정수가 들어있다면?
            # 이때 말하는 것은 number of parameter sets, 내가 넣어보고싶은 파라미터 집합 자체의 개수
            # [ [θ₁₁, θ₁₂, ..., θ₁q],  ← 1번째 weight set
            #   [θ₂₁, θ₂₂, ..., θ₂q],  ← 2번째 weight set
            #   ...
            #   [θM₁, θM₂, ..., θMq]   ← M번째 weight set
            # ]
        else:
            weight_samples = np.asarray(weight_samples) # asarray() 쓰는 이유는, 자료형 통일 
            if len(weight_samples.shape) != 2 or weight_samples.shape[1] != self._model.num_weights:
                raise QiskitMachineLearningError(
                    f"The Effective Dimension class expects"
                    f" a weight_samples array of shape (M, qnn.num_weights)."
                    f" Got {weight_samples.shape}."
                )
            self._weight_samples = weight_samples

        self._num_weight_samples = len(self._weight_samples)

    @property
    def input_samples(self) -> np.ndarray:
        return self._input_samples

    @input_samples.setter
    def input_samples(self, input_samples: Union[np.ndarray, int]) -> None:
        if isinstance(input_samples, int):
            self._input_samples = algorithm_globals.random.normal(
                0, 1, size=(input_samples, self._model.num_inputs)
            )
            # num_inputs는 input의 차원이 되겠지.
            # X = x1, x2, x3 라면 num_inputs는 3
            # 그러면 num_inputs == feature 인거지
            # input_samples가 바로 input dataset의 개수가 된다.
            #[
            #    [x₁₁, x₁₂, x₁₃],  ← 1번째 input
            #    [x₂₁, x₂₂, x₂₃],  ← 2번째 input
            #    ...
            #    [x₁₀₀₁, x₁₀₀₂, x₁₀₀₃]  ← 100번째 input
            #]
        else:
            input_samples = np.asarray(input_samples)
            if len(input_samples.shape) != 2 or input_samples.shape[1] != self._model.num_inputs:
                raise QiskitMachineLearningError(
                    f"The Effective Dimension class expects"
                    f" an input sample array of shape (N, qnn.num_inputs)."
                    f" Got {input_samples.shape}."
                )
            self._input_samples = input_samples

        self._num_input_samples = len(self._input_samples)

    def run_monte_carlo(self) -> Tuple[np.ndarray, np.ndarray]:
        grads: Any = np.zeros(
            (
                self._num_input_samples * self._num_weight_samples,
                self._model.output_shape[0],
                self._model.num_weights,
            )
        )
        outputs: Any = np.zeros(
            (self._num_input_samples * self._num_weight_samples, self._model.output_shape[0])
        )

        for i, param_set in enumerate(self._weight_samples):
            t_before_forward = time.time()
            forward_pass = np.asarray(
                self._model.forward(input_data=self._input_samples, weights=param_set)
            )
            t_after_forward = time.time()

            backward_pass = np.asarray(
                self._model.backward(input_data=self._input_samples, weights=param_set)[1]
            )
            t_after_backward = time.time()

            t_forward = t_after_forward - t_before_forward
            t_backward = t_after_backward - t_after_forward
            logger.debug(
                "Weight sample: %d, forward time: %.3f (s), backward time: %.3f (s)",
                i,
                t_forward,
                t_backward,
            )

            grads[self._num_input_samples * i : self._num_input_samples * (i + 1)] = backward_pass
            outputs[self._num_input_samples * i : self._num_input_samples * (i + 1)] = forward_pass

        if isinstance(self._model, EstimatorQNN):
            grads = np.concatenate([grads / 2, -1 * grads / 2], 1)
            outputs = np.concatenate([(outputs + 1) / 2, (1 - outputs) / 2], 1)

        return grads, outputs

    def get_fisher_information(
        self, gradients: np.ndarray, model_outputs: np.ndarray
    ) -> np.ndarray:

        if model_outputs.shape < gradients.shape:
            model_outputs = np.expand_dims(model_outputs, axis=2)

        gradvectors = np.sqrt(model_outputs) * gradients / model_outputs

        fisher_information = np.einsum("ijk,lji->ikl", gradvectors, gradvectors.T)

        return fisher_information

    def get_normalized_fisher(self, normalized_fisher: np.ndarray) -> Tuple[np.ndarray, float]:

        fisher_trace = np.trace(np.average(normalized_fisher, axis=0))

        fisher_avg = np.average(
            np.reshape(
                normalized_fisher,
                (
                    self._num_weight_samples,
                    self._num_input_samples,
                    self._model.num_weights,
                    self._model.num_weights,
                ),
            ),
            axis=1,
        )

        normalized_fisher = self._model.num_weights * fisher_avg / fisher_trace
        return normalized_fisher, fisher_trace

    def _get_effective_dimension(
        self,
        normalized_fisher: np.ndarray,
        dataset_size: Union[List[int], np.ndarray, int],
    ) -> Union[np.ndarray, int]:
        if not isinstance(dataset_size, int) and len(dataset_size) > 1:
            normalized_fisher = np.expand_dims(normalized_fisher, axis=0)
            n_expanded = np.expand_dims(np.asarray(dataset_size), axis=(1, 2, 3))
            logsum_axis = 1
        else:
            n_expanded = np.asarray(dataset_size)
            logsum_axis = None

        f_mod = normalized_fisher * n_expanded / (2 * np.pi * np.log(n_expanded))
        one_plus_fmod = np.eye(self._model.num_weights) + f_mod
        dets = np.linalg.slogdet(one_plus_fmod)[1]
        dets_div = dets / 2
        effective_dims = (
            2
            * (logsumexp(dets_div, axis=logsum_axis) - np.log(self._num_weight_samples))
            / np.log(dataset_size / (2 * np.pi * np.log(dataset_size)))
        )

        return np.squeeze(effective_dims)

    def get_effective_dimension(
        self, dataset_size: Union[List[int], np.ndarray, int]
    ) -> Union[np.ndarray, int]:

        grads, output = self.run_monte_carlo()
        fisher = self.get_fisher_information(gradients=grads, model_outputs=output)
        normalized_fisher, _ = self.get_normalized_fisher(fisher)
        effective_dimensions = self._get_effective_dimension(normalized_fisher, dataset_size)

        return effective_dimensions


class LocalEffectiveDimension(EffectiveDimension):
    @property
    def weight_samples(self) -> np.ndarray:
        return self._weight_samples

    @weight_samples.setter
    def weight_samples(self, weight_samples: Union[np.ndarray, int]) -> None:
        if isinstance(weight_samples, int):
            self._weight_samples = algorithm_globals.random.uniform(
                0, 1, size=(1, self._model.num_weights)
            )
        else:
            weights = np.asarray(weight_samples)

            if len(weights.shape) < 2:
                weights = np.expand_dims(weight_samples, 0)
            if weights.shape[0] != 1 or weights.shape[1] != self._model.num_weights:
                raise QiskitMachineLearningError(
                    f"The Local Effective Dimension class expects"
                    f" a weight_samples array of shape (1, qnn.num_weights) or (qnn.num_weights)."
                    f" Got {weights.shape}."
                )
            self._weight_samples = weights

        self._num_weight_samples = 1
