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

from contextlib import contextmanager
import torch
from torch.autograd.variable import Variable
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import get_args
from megatron import get_num_microbatches
from megatron import get_timers
from megatron import mpu
import megatron.new_p2p_communication as p2p_communication
from megatron.utils import unwrap_model
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module
from megatron.model import ModelType

def get_nnew_forward_backward_func():
    return forward_backward_pipelining_with_interleaving

def free_output_tensor(output_tensors, deallocate_pipeline_outputs):
    '''Pseudo-free (i.e., set to scalar) the output tensor's '.data' field.

    This method should be called right after the output tensor has been
    sent to the next pipeline stage. At this point, the output tensor is
    only useful for its '.grad_fn' field, and not its '.data'.
    '''
    if not deallocate_pipeline_outputs or output_tensors is None:
        return
    if isinstance(output_tensors, torch.Tensor):
        output_tensors = [output_tensors]
    for output_tensor in output_tensors:
        output_tensor.data = torch.cuda.FloatTensor([0])
        
def custom_backward(output, grad_output):
    '''Directly call C++ autograd engine.

    To make the 'free_output_tensor' (above) optimization work, the C++
    autograd engine must be called directly, bypassing Pytorch's
    torch.autograd.backward. Pytorch's 'backward' checks that the output and
    grad have the same shape, while C++'s 'backward' does not.
    '''

    assert output.numel() == 1, \
        "output should be pseudo-'freed' in schedule, to optimize memory"
    assert isinstance(output, torch.Tensor), \
        "output == '%s'." % type(output).__name__
    assert isinstance(grad_output, (torch.Tensor, type(None))), \
        "grad_output == '%s'." % type(grad_output).__name__

    # Handle scalar output
    if grad_output is None:
        assert output.numel() == 1, "implicit grad requires scalar output."
        grad_output = torch.ones_like(
            output,
            memory_format = torch.preserve_format,
        )

    # Call c++ engine [ see torch/csrc/autograd/python_engine.cpp ]
    Variable._execution_engine.run_backward(
        tensors = (output,),
        grad_tensors = (grad_output,),
        keep_graph = False,
        create_graph = False,
        inputs = tuple(),
        allow_unreachable=True,
        accumulate_grad=True,
    )

def forward_step(forward_step_func, optimizer, data_iterator, model, input_tensor, losses_reduced, model_chunk_id=0):
    """Forward step for passed-in model.

    If first stage, input tensor is obtained from data_iterator, otherwise
    passed-in input_tensor is used.

    Returns output tensor."""
    args = get_args()
    timers = get_timers()

    timers('forward-compute').start()
    unwrapped_model = unwrap_model(
        model, (torchDDP, LocalDDP, Float16Module))

    unwrap_output_tensor = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_output_tensor = True

    unwrapped_model.set_input_tensor(input_tensor)
    lm = unwrapped_model.language_model
    if model_chunk_id == (args.model_chunk_size - 1):
        lm.detach_output = True
    output_tensor, loss_func = forward_step_func(data_iterator, model)
    if model_chunk_id == (args.model_chunk_size - 1):
        output_tensor = loss_func(output_tensor)
        loss, loss_reduced = output_tensor
        backward_step(optimizer, lm.detached_encoder_output, loss / get_num_microbatches(), None)
        output_tensor = lm.encoder_output, lm.detached_encoder_output.grad
        losses_reduced.append(loss_reduced)
    timers('forward-compute').stop()

    if unwrap_output_tensor:
        return output_tensor
    return [output_tensor]


def backward_step(optimizer, input_tensor, output_tensor, output_tensor_grad):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage)."""

    # NOTE: This code currently can handle at most one skip connection. It
    # needs to be modified slightly to support arbitrary numbers of skip
    # connections.
    args = get_args()

    timers = get_timers()
    timers('backward-compute').start()

    # Retain the grad on the input_tensor.
    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True
    for x in input_tensor:
        if x is not None:
            x.retain_grad()

    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    # Backward pass.
    if output_tensor_grad[0] is None:
        output_tensor = optimizer.scale_loss(output_tensor[0])
    if args.deallocate_pipeline_outputs:
        custom_backward(output_tensor[0], output_tensor_grad[0])
    else:
        torch.autograd.backward(output_tensor[0],
                                grad_tensors=output_tensor_grad[0])

    # Collect the grad of the input_tensor.
    input_tensor_grad = [None]
    if input_tensor is not None:
        input_tensor_grad = []
        for x in input_tensor:
            if x is None:
                input_tensor_grad.append(None)
            else:
                input_tensor_grad.append(x.grad)

    if unwrap_input_tensor_grad:
        input_tensor_grad = input_tensor_grad[0]

    timers('backward-compute').stop()

    return input_tensor_grad


def forward_backward_pipelining_with_interleaving(forward_step_func, data_iterator, model,
                                                  optimizer, timers, forward_only):
    """Run interleaved 1F1B schedule (model split into model chunks), with
    communication between pipeline stages as needed.

    Returns dictionary with losses if the last stage, empty dict otherwise."""
    input_tensors = [[] for _ in range(len(model))]
    output_tensors = [[] for _ in range(len(model))]
    losses_reduced = []
    output_tensor_grads = [[] for _ in range(len(model))]

    args = get_args()
    tensor_shape = (args.seq_length, args.micro_batch_size, args.hidden_size)

    # Compute number of warmup and remaining microbatches.
    num_model_chunks = len(model)
    num_microbatches = get_num_microbatches() * num_model_chunks

    def get_batch_id(microbatch_id):
        return microbatch_id % get_num_microbatches()

    def get_model_chunk_id(microbatch_id, forward):
        """Helper method to get the model chunk ID given the iteration number."""
        model_chunk_id = microbatch_id // get_num_microbatches()
        if not forward:
            model_chunk_id = (num_model_chunks - model_chunk_id - 1)
        return model_chunk_id

    def forward_step_helper(microbatch_id):
        """Helper method to run forward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        forward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=True)

        # forward step
        if model_chunk_id == 0:
            if len(input_tensors[model_chunk_id]) == \
                    len(output_tensors[model_chunk_id]):
                input_tensors[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id][get_batch_id(microbatch_id)]
        output_tensor = forward_step(forward_step_func, optimizer,
                                     data_iterator[model_chunk_id],
                                     model[model_chunk_id],
                                     input_tensor, losses_reduced,
                                     model_chunk_id=model_chunk_id)
        if model_chunk_id == num_model_chunks - 1:
            output_tensor, output_tensor_grad = output_tensor
            output_tensor_grads[model_chunk_id].append(output_tensor_grad)
        output_tensors[model_chunk_id].append(output_tensor)

        return output_tensor

    def backward_step_helper(microbatch_id):
        """Helper method to run backward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        backward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=False)

        if model_chunk_id == num_model_chunks - 1:
            if len(output_tensor_grads[model_chunk_id]) == 0:
                output_tensor_grads[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id].pop()
        output_tensor = output_tensors[model_chunk_id].pop()
        output_tensor_grad = output_tensor_grads[model_chunk_id].pop()
        input_tensor_grad = \
            backward_step(optimizer,
                          input_tensor,
                          output_tensor,
                          output_tensor_grad)

        return input_tensor_grad

    for k in range(num_microbatches):
        output_tensor = forward_step_helper(k)

        # Determine if tensor should be received from previous stage.
        forward_model_chunk_id = get_model_chunk_id(k, forward=True)
        if forward_model_chunk_id < num_model_chunks - 1:
            input_tensor = output_tensor.detach().requires_grad_()
            input_tensors[forward_model_chunk_id + 1].append(input_tensor)

    for k in range(num_microbatches):
        input_tensor_grad = backward_step_helper(k)

        backward_model_chunk_id = get_model_chunk_id(k, forward=False)
        if backward_model_chunk_id > 0:
            output_tensor_grads[backward_model_chunk_id - 1].insert(
                0, input_tensor_grad)

    return losses_reduced
