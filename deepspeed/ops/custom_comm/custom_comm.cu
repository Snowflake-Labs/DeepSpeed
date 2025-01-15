#include "custom_comm.h"
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <vector>
#include "tops_context.h"

void ds_create_comm(std::vector<int>& comm_ranks, int rank)
{
    TOPSContext::Instance().create_comm_group(comm_ranks, rank);
}

void ds_create_nccl_comm(std::vector<int>& comm_ranks, int rank, torch::Tensor& nccl_uid)
{
    TOPSContext::Instance().create_nccl_comm(comm_ranks, rank, nccl_uid.data_ptr());
}

void ds_allreduce(torch::Tensor& send_buf, torch::Tensor& rcv_buf, int size, bool async_op)
{
    if (async_op) TOPSContext::Instance().SynchComp();
    ncclAllReduce(send_buf.data_ptr(),
                  rcv_buf.data_ptr(),
                  size,
                  (send_buf.scalar_type() == at::kFloat ? ncclFloat : (send_buf.scalar_type() == at::kHalf ? ncclHalf : ncclBfloat16)),
                  ncclSum,
                  TOPSContext::Instance().GetNCCLComm(),
                  TOPSContext::Instance().GetCommStream());
}

void ds_allgather(torch::Tensor& send_buf, torch::Tensor& rcv_buf, int size, bool async_op)
{
    if (async_op) TOPSContext::Instance().SynchComp();
    ncclAllGather(send_buf.data_ptr(),
                  rcv_buf.data_ptr(),
                  size,
                  (send_buf.scalar_type() == at::kFloat ? ncclFloat : (send_buf.scalar_type() == at::kHalf ? ncclHalf : ncclBfloat16)),
                  TOPSContext::Instance().GetNCCLComm(),
                  TOPSContext::Instance().GetCommStream());
}

void wait_comm() { TOPSContext::Instance().SynchComm(); }

void ds_broadcast(torch::Tensor& send_buf, torch::Tensor& rcv_buf, int size, bool async_op)
{
    ncclBroadcast(send_buf.data_ptr(),
                  rcv_buf.data_ptr(),
                  size,
                  (send_buf.scalar_type() == at::kFloat ? ncclFloat : (send_buf.scalar_type() == at::kHalf ? ncclHalf : ncclBfloat16)),
                  0,
                  TOPSContext::Instance().GetNCCLComm(),
                  TOPSContext::Instance().GetCommStream());
}

void ds_barrier() { TOPSContext::Instance().barrier(); }

inline size_t wordSize(ncclDataType_t type) {
  switch(type) {
    case ncclChar:
    case ncclUint8:
      return 1;
    case ncclHalf:
    case ncclBfloat16:
      return 2;
    case ncclInt:
    case ncclFloat:
    case ncclUint32:
      return 4;
    case ncclInt64:
    case ncclUint64:
    case ncclDouble:
      return 8;
    default: return 0;
  }
}

void ncclAlltoAll(void* sendbuff, 
                        void* recvbuff,
                        size_t count, 
                        ncclDataType_t type, 
                        const unsigned nRanks,
                        ncclComm_t comm, 
                        cudaStream_t stream) {
                            
  size_t rankOffset = count * wordSize(type);

  ncclGroupStart();
  for (int r=0; r<nRanks; r++) {
    ncclSend(((char*)sendbuff)+r*rankOffset, count, type, r, comm, stream);
    ncclRecv(((char*)recvbuff)+r*rankOffset, count, type, r, comm, stream);
  }
  ncclGroupEnd();
}

void ds_alltoall(torch::Tensor& send_buf, torch::Tensor& rcv_buf, int size, bool async_op)
{
    ncclAlltoAll(send_buf.data_ptr(),
                  rcv_buf.data_ptr(),
                  size,
                  (send_buf.scalar_type() == at::kFloat ? 
                    ncclFloat : 
                    (send_buf.scalar_type() == at::kHalf ? 
                    ncclHalf : 
                    (send_buf.scalar_type() == torch::kInt8 ? ncclUint8 : ncclBfloat16))),
                  TOPSContext::Instance().GetNumRanks(),
                  TOPSContext::Instance().GetNCCLComm(),
                  TOPSContext::Instance().GetCommStream());
}

torch::Tensor ds_get_nccl_uid()
{

  auto options = at::TensorOptions()
                       .dtype(torch::kUInt8)
                       .layout(torch::kStrided)
                       .device(torch::kCPU)
                       .requires_grad(false);
    auto nccl_uid = TOPSContext::Instance().get_nccl_uid();
    auto uid_tensor = torch::from_blob((void*)&nccl_uid, {sizeof(ncclUniqueId)}, options);
    return uid_tensor;
}
