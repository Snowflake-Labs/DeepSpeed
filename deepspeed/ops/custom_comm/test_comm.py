import os
import deepspeed
from deepspeed.ops import create_comm, get_default_comm, Layout
from deepspeed.accelerator import get_accelerator
from deepspeed.comm import init_distributed
import torch
import time

def execute(val, is_torch=True, comm=None, op='AllReduce'):
    if is_torch:
        if op == 'AllReduce':
            torch.distributed.all_reduce(val)
            return val
        elif op == 'AllGather':
            world_size = torch.distributed.get_world_size()
            result = torch.empty((world_size*val.shape[0],) + val.shape[1:])
            torch.distributed.all_gather_into_tensor(result, val)
            return result
        elif op == 'AlltoAll':
            result = torch.empty_like(val)
            torch.distributed.all_to_all_single(result, val)
            return result
    else:
        if op == 'AllReduce':
            val, _ = comm.all_reduce(val)
            return val
        elif op == 'AllGather':
            val, _ = comm.all_gather(val)
            return val
        elif op == 'AlltoAll':
            val, _ = comm.all_to_all(val)
            return val
            
def init_comm():
    init_distributed(dist_backend='nccl')
    global_rank = torch.distributed.get_rank()
    group_stride = 1
    group_size = torch.distributed.get_world_size() // group_stride
    gid = global_rank // group_size
    comm = create_comm(Layout(group_size, group_stride, world_size=torch.distributed.get_world_size()))
    val = torch.arange(group_size, dtype=torch.bfloat16, device=torch.cuda.current_device())
    
    # test to see all is working
    val,_ = comm.all_to_all(val)
    print(f'[{global_rank}]: alltoall -> {val}')
    # exit()
    return comm

def run_nccl_comm_test(comm, op='AllReduce'):
    
    global_rank = torch.distributed.get_rank()

    inp = torch.randn(512*5120, dtype=torch.half, device=torch.cuda.current_device())
    #weight1 = torch.randn(4096, 4096, dtype=torch.half, device=torch.cuda.current_device())
    #weight2 = torch.randn(4096, 16384, dtype=torch.half, device=torch.cuda.current_device())
    
    for _ in range(10):
        #val = torch.matmul(inp.view(-1, 4096), weight1)
        val = execute(inp, is_torch=False, comm=comm, op=op)
        #out = torch.matmul(val.view(-1, 4096), weight2)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        #val = torch.matmul(inp.view(-1, 4096), weight1)
        val = execute(inp, is_torch=False, comm=comm, op=op)
        #out = torch.matmul(val.view(-1, 4096), weight2)
    torch.cuda.synchronize()
    end = time.time()
    ds_time = end - start
    if global_rank == 0:
        print(f'------------------- ds_comm execution time for {op}: {end - start} ms -------------------')
    #print(f'[{global_rank}]: {end - start} ms')

    for _ in range(10):
        #val = torch.matmul(inp.view(-1, 4096), weight1)
        val = execute(inp, is_torch=True, op=op)
        #out = torch.matmul(val.view(-1, 4096), weight2)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        #val = torch.matmul(inp.view(-1, 4096), weight1)
        val = execute(inp, is_torch=True, op=op)
        #out = torch.matmul(val.view(-1, 4096), weight2)
    torch.cuda.synchronize()
    end = time.time()
    pt_time = end - start
    if global_rank == 0:
        print(f'------------------- torch execution time for {op}: {end - start} ms -------------------')
        print(f'speedup: {pt_time / ds_time}x')
    #print(f'[{global_rank}]: {end - start} ms')
    
comm = init_comm()
run_nccl_comm_test(comm=comm, op='AllReduce')
run_nccl_comm_test(comm=comm, op='AlltoAll')
