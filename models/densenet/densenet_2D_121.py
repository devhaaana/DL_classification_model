from .densenet_2D import *


def Model(input_dim, target_dim):
    num_blocks = [6, 12, 24, 16]
    # growth_rate = 32
    # return DenseNet(num_blocks=num_blocks, growth_rate=growth_rate)
    return DenseNet(num_blocks=num_blocks)
