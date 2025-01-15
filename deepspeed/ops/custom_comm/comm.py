import torch
import os
from deepspeed.ops.op_builder import CommBuilder
from deepspeed.comm import init_distributed
from .layout import Layout
ds_comm = None

class Comm:
    current_comm = None

    def __init__(self, layout: Layout, local_rank: int):
        global ds_comm
        if ds_comm is None:
            ds_comm = CommBuilder().load()
            
        self.ds_comm = ds_comm
        self._layout = layout
        self.my_rank = local_rank
        self.global_rank = torch.distributed.get_rank()
        self.group_size = layout._group_size

        self.global_ranks = layout.sibling_ranks(self.global_rank)
        self.local_ranks = list(range(self.group_size))
        self.rank_map = dict(zip(self.global_ranks, self.local_ranks))
        
        print("Initializing comm ...")

    def all_reduce(self, val, inplace=True, async_op=False):
        val_sum = val if inplace else torch.empty_like(val)
        op = communicate_op(val, val_sum, async_op, op_type="all_reduce")
        return val_sum, op

    def all_gather(self, val, inplace=True, async_op=False):
        val_gather = torch.empty((self.group_size * val.size(0),
                                  *val.shape[1:]),
                                 device=val.device,
                                 dtype=val.dtype)
        op = communicate_op(val, val_gather, async_op, op_type="all_gather")
        return val_gather, op

    def all_to_all(self, val, result=None, inplace=True, async_op=False):
        result = result if result is not None else torch.empty_like(val)
        op = communicate_op(val, result, async_op, world_size=self.group_size, op_type="all_to_all")
        return result, op

    def broadcast(self, val, inplace=True, async_op=False):
        val_bcst = torch.empty_like(val)
        op = communicate_op(val, val_bcst, async_op, op_type="broadcast")
        return val_bcst, op

    def barrier(self):
        ds_comm.wait_comm()
        ds_comm.barrier()

    @classmethod
    def get_current_comm(cls):
        if cls.current_comm is None:
            cls.current_comm = NcclComm()
        return cls.current_comm


class communicate_op:
    def __init__(self, val, result, async_op, world_size=None, op_type="all_reduce"):
        if op_type == "all_reduce":
            ds_comm.allReduce(val, result, val.numel(), async_op)
        elif op_type == "all_gather":
            ds_comm.allGather(val, result, val.numel(), async_op)
        elif op_type == "all_to_all":
            ds_comm.alltoall(val, result, val.numel() // world_size, async_op)
        elif op_type == "broadcast":
            ds_comm.broadcast(val, result, val.numel(), async_op)

    def wait(self):
        ds_comm.wait_comm()


def get_default_comm():
    return Comm.get_current_comm()
