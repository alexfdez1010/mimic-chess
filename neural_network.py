from typing import Tuple

import torch
from torch import squeeze, Tensor, softmax
from torch.nn import Module, ReLU, Sequential, Conv2d, \
    BatchNorm2d, AdaptiveAvgPool2d, Sigmoid, Flatten, Linear, \
    init, Parameter

from utils.constants import CHANNELS, ILLEGAL_MOVE_PENALTY, ROWS, COLS
from utils.uci_to_action import create_policy_matrix

FINAL_CHANNELS_POLICY: int = 73  # number of channels in the last convolutional layer of the policy network
FINAL_CHANNELS_VALUE: int = 3  # number of channels in the last convolutional layer of the value network
FINAL_CHANNELS_TIME: int = 3  # number of channels in the last convolutional layer of the time network
RESIDUAL_CHANNELS: int = 64  # number of channels in the residual blocks
NUM_RESIDUAL_BLOCKS: int = 6  # number of residual blocks
NUM_OF_OUTCOMES: int = 3  # number of possible outcomes of a game
CHANNELS_FEATURE_TIME: int = 128  # number of channels for the feature extractor of the time network


class SEBlock(Module):

    def __init__(self, channels, r=8):
        super().__init__()

        self.pooled = AdaptiveAvgPool2d(1)
        self.squeeze = Sequential(
            ReLU(),
            Flatten(),
            Linear(channels, channels // r),
        )
        self.excitation = Linear(channels // r, 2 * channels)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        out = self.pooled(x)
        out = self.squeeze(out)
        out = self.excitation(out)
        out = torch.reshape(out, [-1, out.shape[1], 1, 1])
        gammas, betas = torch.split(out, x.shape[1], dim=1)
        return self.sigmoid(gammas) * x + betas


class ResidualBlock(Module):

    def __init__(self, channels):
        super().__init__()
        self.conv1 = Conv2d(channels, channels, 3, stride=1, padding='same', bias=False)
        self.conv1_bn = CenteredBatchNorm2d(channels)
        self.conv2 = Conv2d(channels, channels, 3, stride=1, padding='same', bias=False)
        self.conv2_bn = CenteredBatchNorm2d(channels)
        self.relu = ReLU()
        self.se = SEBlock(channels)

    def forward(self, x):
        out = self.conv1_bn(self.conv1(x))
        out = self.relu(out)
        out = self.conv2_bn(self.conv2(out))
        out = self.se(out)
        out += x
        out = self.relu(out)
        return out


class ConvBlock(Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1, bias=False)
        self.bn = CenteredBatchNorm2d(out_channels)
        self.relu = ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class CenteredBatchNorm2d(BatchNorm2d):

    def __init__(self, channels):
        super().__init__(channels, affine=True)
        self.weight.data.fill_(1)
        self.weight.requires_grad = False


class FeatureExtractor(Module):

    def __init__(self):
        super().__init__()
        self.conv_block = ConvBlock(CHANNELS, RESIDUAL_CHANNELS)
        self.residual_blocks = [ResidualBlock(RESIDUAL_CHANNELS) for _ in range(NUM_RESIDUAL_BLOCKS)]
        self.body = Sequential(*self.residual_blocks)

    def forward(self, x):
        out = self.conv_block(x)
        out = self.body(out)
        return out


class TimeFeatureExtractor(Module):

    def __init__(self):
        super().__init__()
        self.linear = Linear(3, CHANNELS_FEATURE_TIME)
        self.relu = ReLU()
        self.linear2 = Linear(CHANNELS_FEATURE_TIME, CHANNELS_FEATURE_TIME)
        self.relu2 = ReLU()

    def forward(self, x):
        out = self.linear(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu2(out)

        return out


class PolicyMap(Module):

    def __init__(self):
        super().__init__()
        self.flatten = Flatten()
        self.policy_matrix = Parameter(create_policy_matrix(), requires_grad=False)

    def forward(self, x):
        out = self.flatten(x)
        out = squeeze(out)
        out = out @ self.policy_matrix
        return out


class HeadPolicy(Module):

    def __init__(self):
        super(HeadPolicy, self).__init__()
        self.conv_block_policy = ConvBlock(
            RESIDUAL_CHANNELS + CHANNELS_FEATURE_TIME // (ROWS*COLS), RESIDUAL_CHANNELS, 3
        )
        self.conv_final_policy = Conv2d(RESIDUAL_CHANNELS, FINAL_CHANNELS_POLICY, 1)

        self.policy_map = PolicyMap()
        self.illegal_move_penalty = Parameter(
            torch.tensor(ILLEGAL_MOVE_PENALTY, dtype=torch.float32), requires_grad=False
        )

    def forward(self, x: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        features_position, action_mask, features_times = x
        out = torch.reshape(features_times, [-1, CHANNELS_FEATURE_TIME // (ROWS*COLS), ROWS, COLS])
        out = torch.cat([features_position, out], dim=1)
        out = self.conv_block_policy(out)
        out = self.conv_final_policy(out)
        out = self.policy_map(out)

        out = torch.where(action_mask, out, self.illegal_move_penalty)

        return out


class HeadValue(Module):

    def __init__(self):
        super(HeadValue, self).__init__()
        self.conv_block_value = ConvBlock(RESIDUAL_CHANNELS, RESIDUAL_CHANNELS // 2, 3)
        self.conv_final_value = Conv2d(RESIDUAL_CHANNELS // 2, FINAL_CHANNELS_VALUE, 1)
        self.flatten_value = Flatten()
        self.linear_value = Linear(FINAL_CHANNELS_VALUE * 8 * 8 + CHANNELS_FEATURE_TIME, 3)

    def forward(self, x):
        features_position, features_time = x
        out = self.conv_block_value(features_position)
        out = self.conv_final_value(out)
        out = self.flatten_value(out)
        out = torch.cat([out, features_time], dim=1)
        out = self.linear_value(out)
        return out


class HeadTime(Module):

    def __init__(self):
        super(HeadTime, self).__init__()
        self.conv_block_time = ConvBlock(RESIDUAL_CHANNELS, RESIDUAL_CHANNELS // 2, 3)
        self.conv_final_time = Conv2d(RESIDUAL_CHANNELS // 2, FINAL_CHANNELS_TIME, 1)
        self.flatten_time = Flatten()
        self.linear_time = Linear(FINAL_CHANNELS_TIME * 8 * 8 + CHANNELS_FEATURE_TIME, 1)
        self.sigmoid_time = Sigmoid()

    def forward(self, x):
        features_position, features_time = x
        out = self.conv_block_time(features_position)
        out = self.conv_final_time(out)
        out = self.flatten_time(out)
        out = torch.cat([out, features_time], dim=1)
        out = self.linear_time(out)
        out = self.sigmoid_time(out)
        return out


class NeuralNetwork(Module):
    """Clase que representa a la red neuronal"""

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.feature_extractor = FeatureExtractor()
        self.time_feature_extractor = TimeFeatureExtractor()

        self.policy = HeadPolicy()
        self.value = HeadValue()
        self.time = HeadTime()

        self.apply(self.__init_weights)

    def forward(self, x: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass of the network
        :param x: input of the network (observation, action_mask, times)
        :return: output of the network (policy, value, time)
        """
        observation, action_mask, times = x

        out_extractor = self.feature_extractor(observation)
        out_time = self.time_feature_extractor(times)

        out_policy = self.policy((out_extractor, action_mask, out_time))
        out_value = self.value((out_extractor, out_time))
        out_time = self.time((out_extractor, out_time))

        return out_policy, out_value, out_time

    @staticmethod
    def __init_weights(module: Module) -> None:
        """
        Allows to initialize the weights of the network
        :param module: module to initialize
        """
        if isinstance(module, (Conv2d, Linear)):
            init.xavier_normal_(module.weight, gain=1.0)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, BatchNorm2d):
            module.weight.data.fill_(1)
            module.bias.data.zero_()


class NeuralNetworkWithSoftmax(Module):
    """
    Wrapper for the neural network that applies softmax to the output of the policy and value heads
    """

    def __init__(self, neural_network: NeuralNetwork):
        """
        Creates an instance of the class NeuralNetworkWithSoftmax
        :param neural_network: neural network to wrap
        """
        super().__init__()
        self.neural_network = neural_network

    def forward(self, x):
        """Overrides the forward method of the Module class and returns the output of the neural network"""
        policy, value, time = self.neural_network(x)
        policy = softmax(policy, dim=1)
        value = softmax(value, dim=1)
        return policy, value, time

    def state_dict(self, *args, **kwargs):
        """Overrides the state_dict method to return the state of the neural network"""
        return self.neural_network.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        """Overrides the load_state_dict method to load the state of the neural network"""
        return self.neural_network.load_state_dict(*args, **kwargs)
