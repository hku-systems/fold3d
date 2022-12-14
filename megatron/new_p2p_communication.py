# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import reduce
import operator
import torch

from megatron import get_args
from megatron import mpu


def _gather(tensor_recv):
    args = get_args()
    tensor_shape = (args.seq_length, args.micro_batch_size, args.hidden_size)

    if args.scatter_gather_tensors_in_pipeline:
        tensor_recv = mpu.gather_split_1d_tensor(
            tensor_recv).view(tensor_shape).requires_grad_()
    return tensor_recv


def send_helper_thread(queue, num_microbatches, rank, group):
    for i in range(num_microbatches):
        tensor_send_cpu = queue.get()
        torch.distributed.send(tensor_send_cpu, rank, group=group)


def _tensor_chunk_shape():
    args = get_args()
    tensor_shape = (args.seq_length, args.micro_batch_size, args.hidden_size)
    if args.scatter_gather_tensors_in_pipeline:
        tensor_chunk_shape = reduce(operator.mul, tensor_shape, 1)
        if tensor_chunk_shape % mpu.get_tensor_model_parallel_world_size() == 0:
            tensor_chunk_shape = tensor_chunk_shape // \
                mpu.get_tensor_model_parallel_world_size()
        else:
            tensor_chunk_shape = tensor_shape
    else:
        tensor_chunk_shape = tensor_shape
    return tensor_chunk_shape


def recv_helper_thread(queue, local_rank, stream, tensor_chunk_shape, dtype, num_microbatches, rank, group):
    torch.cuda.set_device(local_rank)
    for i in range(num_microbatches):
        tensor_recv_cpu = torch.empty(tensor_chunk_shape, dtype=dtype, device='cpu', pin_memory=True)
        torch.distributed.recv(tensor_recv_cpu, rank, group=group)
        with torch.cuda.stream(stream):
            tensor_recv = tensor_recv_cpu.cuda()
            e = torch.cuda.Event()
            e.record()
        queue.put((tensor_recv, e,))


def _communicate(tensor_send_next, tensor_send_prev, recv_prev, recv_next,
                 tensor_shape,
                 use_ring_exchange=False,
                 dtype_=None):
    """Communicate tensors between stages. Used as helper method in other
    communication methods that are used in megatron/schedules.py.

    Takes the following arguments:
        tensor_send_next: tensor to send to next rank (no tensor sent if
                          set to None).
        tensor_send_prev: tensor to send to prev rank (no tensor sent if
                          set to None).
        recv_prev: boolean for whether tensor should be received from
                   previous rank.
        recv_next: boolean for whether tensor should be received from
                   next rank.
        tensor_shape: shape of tensor to receive (this method assumes that all
                      tensors sent and received in a single function call are
                      the same shape).
        use_ring_exchange: boolean for whether torch.distributed.ring_exchange()
                           API should be used.
        dtype_: optional, this is used when the tensor that needs to be
                communicated is different from args.params_dtype.
    Returns:
        (tensor_recv_prev, tensor_recv_next)
    """
    args = get_args()

    # Create placeholder tensors for receive in forward and backward directions
    # if needed.
    tensor_recv_prev = None
    tensor_recv_next = None

    # Some legacy inference code doesn't set the tensor shape, do so now
    # for the normal values for gpt/bert. This could be removed if inference
    # code is changed to provide tensor_shape.
    if tensor_shape is None:
        tensor_shape = (args.seq_length, args.micro_batch_size, args.hidden_size)

    override_scatter_gather_tensors_in_pipeline = False
    if args.scatter_gather_tensors_in_pipeline:
        tensor_chunk_shape = reduce(operator.mul, tensor_shape, 1)
        if tensor_chunk_shape % mpu.get_tensor_model_parallel_world_size() == 0:
            tensor_chunk_shape = tensor_chunk_shape // \
                mpu.get_tensor_model_parallel_world_size()
        else:
            tensor_chunk_shape = tensor_shape
            override_scatter_gather_tensors_in_pipeline = True
    else:
        tensor_chunk_shape = tensor_shape
    dtype = args.params_dtype
    if args.fp32_residual_connection:
        dtype = torch.float

    requires_grad = True
    if dtype_ is not None:
        dtype = dtype_
        requires_grad = False

    device = torch.cuda.current_device()
    if recv_prev:
        tensor_recv_prev = torch.empty(tensor_chunk_shape,
                                       requires_grad=requires_grad,
                                       device=device,
                                       dtype=dtype)
    if recv_next:
        tensor_recv_next = torch.empty(tensor_chunk_shape,
                                       requires_grad=requires_grad,
                                       device=device,
                                       dtype=dtype)

    # Split tensor into smaller chunks if using scatter-gather optimization.
    if not override_scatter_gather_tensors_in_pipeline and \
            args.scatter_gather_tensors_in_pipeline:
        if tensor_send_next is not None:
            tensor_send_next = mpu.split_tensor_into_1d_equal_chunks(tensor_send_next)

        if tensor_send_prev is not None:
            tensor_send_prev = mpu.split_tensor_into_1d_equal_chunks(tensor_send_prev)

    if args.use_gloo_for_pipeline_parallel_send_recv:
        if tensor_send_prev is not None:
            tensor_send_prev_cpu = torch.empty_like(tensor_send_prev, device='cpu', pin_memory=True)
            tensor_send_prev_cpu.copy_(tensor_send_prev)
            return tensor_send_prev_cpu
        if tensor_send_next is not None:
            tensor_send_next_cpu = torch.empty_like(tensor_send_next, device='cpu', pin_memory=True)
            tensor_send_next_cpu.copy_(tensor_send_next)
            return tensor_send_next_cpu

    # Send tensors in both the forward and backward directions as appropriate.
    if tensor_send_prev is not None:
        if args.use_gloo_for_pipeline_parallel_broadcast:
            return torch.distributed.broadcast(tensor_send_prev,
                mpu.get_pipeline_model_parallel_current_rank(),
                group=mpu.get_pipeline_model_parallel_prev_rank_group(),
                async_op=True)
        return torch.distributed.isend(tensor_send_prev,
            mpu.get_pipeline_model_parallel_prev_rank(),
            group=mpu.get_pipeline_model_parallel_prev_rank_group())
    if tensor_send_next is not None:
        if args.use_gloo_for_pipeline_parallel_broadcast:
            return torch.distributed.broadcast(tensor_send_next,
                mpu.get_pipeline_model_parallel_current_rank(),
                group=mpu.get_pipeline_model_parallel_next_rank_group(),
                async_op=True)
        return torch.distributed.isend(tensor_send_next,
            mpu.get_pipeline_model_parallel_next_rank(),
            group=mpu.get_pipeline_model_parallel_next_rank_group())
    if tensor_recv_prev is not None:
        if args.use_gloo_for_pipeline_parallel_broadcast:
            return tensor_recv_prev, torch.distributed.broadcast(tensor_recv_prev,
                mpu.get_pipeline_model_parallel_prev_rank(),
                group=mpu.get_pipeline_model_parallel_prev_rank_group(),
                async_op=True)
        return tensor_recv_prev, torch.distributed.irecv(tensor_recv_prev,
            mpu.get_pipeline_model_parallel_prev_rank(),
            group=mpu.get_pipeline_model_parallel_prev_rank_group())
    if tensor_recv_next is not None:
        if args.use_gloo_for_pipeline_parallel_broadcast:
            return tensor_recv_next, torch.distributed.broadcast(tensor_recv_next,
                mpu.get_pipeline_model_parallel_next_rank(),
                group=mpu.get_pipeline_model_parallel_next_rank_group(),
                async_op=True)
        return tensor_recv_next, torch.distributed.irecv(tensor_recv_next,
            mpu.get_pipeline_model_parallel_next_rank(),
            group=mpu.get_pipeline_model_parallel_next_rank_group())

    # if len(ops) > 0:
    #     reqs = torch.distributed.batch_isend_irecv(ops)
    #     for req in reqs:
    #         req.wait()
    # To protect against race condition when using batch_isend_irecv().
    # torch.cuda.synchronize()

    # If using scatter-gather optimization, gather smaller chunks.
    # if not override_scatter_gather_tensors_in_pipeline and \
    #         args.scatter_gather_tensors_in_pipeline:
    #     if recv_prev:
    #         tensor_recv_prev = mpu.gather_split_1d_tensor(
    #             tensor_recv_prev).view(tensor_shape).requires_grad_()

    #     if recv_next:
    #         tensor_recv_next = mpu.gather_split_1d_tensor(
    #             tensor_recv_next).view(tensor_shape).requires_grad_()

    # return tensor_recv_prev, tensor_recv_next


def recv_forward(tensor_shape=None, dtype_=None, timers=None):
    """Receive tensor from previous rank in pipeline (forward receive)."""

    return _communicate(
        tensor_send_next=None,
        tensor_send_prev=None,
        recv_prev=True,
        recv_next=False,
        tensor_shape=tensor_shape,
        dtype_=dtype_)


def recv_backward(tensor_shape=None, timers=None):
    """Receive tensor from next rank in pipeline (backward receive)."""
    return _communicate(
        tensor_send_next=None,
        tensor_send_prev=None,
        recv_prev=False,
        recv_next=True,
        tensor_shape=tensor_shape)


def send_forward(output_tensor, tensor_shape=None, dtype_=None, timers=None):
    """Send tensor to next rank in pipeline (forward send)."""

    if not mpu.is_pipeline_last_stage():
        return _communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=False,
            tensor_shape=tensor_shape,
            dtype_=dtype_)


def send_backward(input_tensor_grad, tensor_shape=None, timers=None):
    """Send tensor to previous rank in pipeline (backward send)."""
    if not mpu.is_pipeline_first_stage():
        return _communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=False,
            recv_next=False,
            tensor_shape=tensor_shape)


def send_forward_recv_backward(output_tensor, tensor_shape=None, timers=None):
    """Batched send and recv with next rank in pipeline."""
    if mpu.is_pipeline_last_stage():
        output_tensor_grad = None
    else:
        if timers is not None:
            timers('forward-send-backward-recv').start()
        _, output_tensor_grad = _communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            tensor_shape=tensor_shape)
        if timers is not None:
            timers('forward-send-backward-recv').stop()
    return output_tensor_grad


def send_backward_recv_forward(input_tensor_grad, tensor_shape=None, timers=None):
    """Batched send and recv with previous rank in pipeline."""
    if mpu.is_pipeline_first_stage():
        input_tensor = None
    else:
        if timers is not None:
            timers('backward-send-forward-recv').start()
        input_tensor, _ = _communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=True,
            recv_next=False,
            tensor_shape=tensor_shape)
        if timers is not None:
            timers('backward-send-forward-recv').stop()
    return input_tensor


def send_forward_recv_forward(output_tensor, recv_prev, tensor_shape=None, timers=None):
    """Batched recv from previous rank and send to next rank in pipeline."""
    if timers is not None:
        timers('forward-send-forward-recv').start()
    input_tensor, _ = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=None,
        recv_prev=recv_prev,
        recv_next=False,
        tensor_shape=tensor_shape)
    if timers is not None:
        timers('forward-send-forward-recv').stop()
    return input_tensor


def send_backward_recv_backward(input_tensor_grad, recv_next, tensor_shape=None, timers=None):
    """Batched recv from next rank and send to previous rank in pipeline."""
    if timers is not None:
        timers('backward-send-backward-recv').start()
    _, output_tensor_grad = _communicate(
        tensor_send_next=None,
        tensor_send_prev=input_tensor_grad,
        recv_prev=False,
        recv_next=recv_next,
        tensor_shape=tensor_shape)
    if timers is not None:
        timers('backward-send-backward-recv').stop()
    return output_tensor_grad


def send_forward_backward_recv_forward_backward(
        output_tensor, input_tensor_grad, recv_prev,
        recv_next, tensor_shape=None, timers=None):
    """Batched send and recv with previous and next ranks in pipeline."""
    if timers is not None:
        timers('forward-backward-send-forward-backward-recv').start()
    input_tensor, output_tensor_grad = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=input_tensor_grad,
        recv_prev=recv_prev,
        recv_next=recv_next,
        tensor_shape=tensor_shape)
    if timers is not None:
        timers('forward-backward-send-forward-backward-recv').stop()
    return input_tensor, output_tensor_grad
