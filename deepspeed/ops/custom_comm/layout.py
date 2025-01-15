import torch
import os
from deepspeed.ops.op_builder import CommBuilder
from deepspeed.comm import init_distributed

class Layout:
    def __init__(self, g_size=None, stride=1, world_size=1):

        num_groups = world_size // g_size
        self._group_size = g_size
        self._sibling_ranks = []
        for gid in range(num_groups):
            if stride == 1:
                self._sibling_ranks.append(
                    [gid * g_size + i for i in range(g_size)]
                )
            else:
                self._sibling_ranks.append(
                    [gid % stride + i * stride for i in range(g_size)]
                )

    def sibling_ranks(self, rank):
        for sranks in self._sibling_ranks:
            if rank in sranks:
                break
        return sranks
    
    def parent_rank(self, rank):
        return self.sibling_ranks(rank)[0]
