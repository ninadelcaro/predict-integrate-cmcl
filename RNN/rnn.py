from os.path import exists, join
from os import mkdir
import json
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from typing import Tuple, Optional, overload
from torch import Tensor
from torch.nn import Module
from torch.nn import Parameter
from torch.nn.utils.rnn import PackedSequence
from torch.nn import init
from torch import _VF

class RNN(nn.RNNBase):
    r"""__init__(input_size,hidden_size,num_layers=1,nonlinearity='tanh',bias=True,batch_first=False,dropout=0.0,bidirectional=False,device=None,dtype=None)

    Apply a multi-layer Elman RNN with :math:`\tanh` or :math:`\text{ReLU}`
    non-linearity to an input sequence. For each element in the input sequence,
    each layer computes the following function:

    .. math::
        h_t = \tanh(x_t W_{ih}^T + b_{ih} + h_{t-1}W_{hh}^T + b_{hh})

    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is
    the input at time `t`, and :math:`h_{(t-1)}` is the hidden state of the
    previous layer at time `t-1` or the initial hidden state at time `0`.
    If :attr:`nonlinearity` is ``'relu'``, then :math:`\text{ReLU}` is used instead of :math:`\tanh`.

    .. code-block:: python

        # Efficient implementation equivalent to the following with bidirectional=False
        def forward(x, h_0=None):
            if batch_first:
                x = x.transpose(0, 1)
            seq_len, batch_size, _ = x.size()
            if h_0 is None:
                h_0 = torch.zeros(num_layers, batch_size, hidden_size)
            h_t_minus_1 = h_0
            h_t = h_0
            output = []
            for t in range(seq_len):
                for layer in range(num_layers):
                    h_t[layer] = torch.tanh(
                        x[t] @ weight_ih[layer].T
                        + bias_ih[layer]
                        + h_t_minus_1[layer] @ weight_hh[layer].T
                        + bias_hh[layer]
                    )
                output.append(h_t[-1])
                h_t_minus_1 = h_t
            output = torch.stack(output)
            if batch_first:
                output = output.transpose(0, 1)
            return output, h_t

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two RNNs together to form a `stacked RNN`,
            with the second RNN taking in outputs of the first RNN and
            computing the final results. Default: 1
        nonlinearity: The non-linearity to use. Can be either ``'tanh'`` or ``'relu'``. Default: ``'tanh'``
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        batch_first: If ``True``, then the input and output tensors are provided
            as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
            Note that this does not apply to hidden or cell states. See the
            Inputs/Outputs sections below for details.  Default: ``False``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            RNN layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``

    Inputs: input, h_0
        * **input**: tensor of shape :math:`(L, H_{in})` for unbatched input,
          :math:`(L, N, H_{in})` when ``batch_first=False`` or
          :math:`(N, L, H_{in})` when ``batch_first=True`` containing the features of
          the input sequence.  The input can also be a packed variable length sequence.
          See :func:`torch.nn.utils.rnn.pack_padded_sequence` or
          :func:`torch.nn.utils.rnn.pack_sequence` for details.
        * **h_0**: tensor of shape :math:`(D * \text{num\_layers}, H_{out})` for unbatched input or
          :math:`(D * \text{num\_layers}, N, H_{out})` containing the initial hidden
          state for the input sequence batch. Defaults to zeros if not provided.

        where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                L ={} & \text{sequence length} \\
                D ={} & 2 \text{ if bidirectional=True otherwise } 1 \\
                H_{in} ={} & \text{input\_size} \\
                H_{out} ={} & \text{hidden\_size}
            \end{aligned}

    Outputs: output, h_n
        * **output**: tensor of shape :math:`(L, D * H_{out})` for unbatched input,
          :math:`(L, N, D * H_{out})` when ``batch_first=False`` or
          :math:`(N, L, D * H_{out})` when ``batch_first=True`` containing the output features
          `(h_t)` from the last layer of the RNN, for each `t`. If a
          :class:`torch.nn.utils.rnn.PackedSequence` has been given as the input, the output
          will also be a packed sequence.
        * **h_n**: tensor of shape :math:`(D * \text{num\_layers}, H_{out})` for unbatched input or
          :math:`(D * \text{num\_layers}, N, H_{out})` containing the final hidden state
          for each element in the batch.

    Attributes:
        weight_ih_l[k]: the learnable input-hidden weights of the k-th layer,
            of shape `(hidden_size, input_size)` for `k = 0`. Otherwise, the shape is
            `(hidden_size, num_directions * hidden_size)`
        weight_hh_l[k]: the learnable hidden-hidden weights of the k-th layer,
            of shape `(hidden_size, hidden_size)`
        bias_ih_l[k]: the learnable input-hidden bias of the k-th layer,
            of shape `(hidden_size)`
        bias_hh_l[k]: the learnable hidden-hidden bias of the k-th layer,
            of shape `(hidden_size)`

    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`

    .. note::
        For bidirectional RNNs, forward and backward are directions 0 and 1 respectively.
        Example of splitting the output layers when ``batch_first=False``:
        ``output.view(seq_len, batch, num_directions, hidden_size)``.

    .. note::
        ``batch_first`` argument is ignored for unbatched inputs.

    .. include:: ../cudnn_rnn_determinism.rst

    .. include:: ../cudnn_persistent_rnn.rst

    Examples::

        >>> rnn = nn.RNN(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> output, hn = rnn(input, h0)
    """

    @overload
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 nonlinearity: str = 'tanh', bias: bool = True, batch_first: bool = False,
                 dropout: float = 0., bidirectional: bool = False, device=None,
                 dtype=None) -> None:
        ...

    @overload
    def __init__(self, *args, **kwargs):
        ...

    def __init__(self, *args, **kwargs):
        if 'proj_size' in kwargs:
            raise ValueError("proj_size argument is only supported for LSTM, not RNN or GRU")
        if len(args) > 3:
            self.nonlinearity = args[3]
            args = args[:3] + args[4:]
        else:
            self.nonlinearity = kwargs.pop('nonlinearity', 'tanh')
        if self.nonlinearity == 'tanh':
            mode = 'RNN_TANH'
        elif self.nonlinearity == 'relu':
            mode = 'RNN_RELU'
        else:
            raise ValueError(f"Unknown nonlinearity '{self.nonlinearity}'. Select from 'tanh' or 'relu'.")
        super().__init__(mode, *args, **kwargs)

    @overload
    @torch._jit_internal._overload_method  # noqa: F811
    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        pass

    @overload
    @torch._jit_internal._overload_method  # noqa: F811
    def forward(self, input: PackedSequence, hx: Optional[Tensor] = None) -> Tuple[PackedSequence, Tensor]:
        pass

    def forward(self, input, hx=None):  # noqa: F811
        print("hello there!")
        self._init_flat_weights()

        num_directions = 2 if self.bidirectional else 1
        orig_input = input

        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            # script() is unhappy when max_batch_size is different type in cond branches, so we duplicate
            if hx is None:
                hx = torch.zeros(self.num_layers * num_directions,
                                 max_batch_size, self.hidden_size,
                                 dtype=input.dtype, device=input.device)
                print("input is packed, hx is None")
            else:
                # Each batch of the hidden state should match the input sequence that
                # the user believes he/she is passing in.
                hx = self.permute_hidden(hx, sorted_indices)
        else:
            batch_sizes = None
            if input.dim() not in (2, 3):
                raise ValueError(f"RNN: Expected input to be 2D or 3D, got {input.dim()}D tensor instead")
            is_batched = input.dim() == 3
            batch_dim = 0 if self.batch_first else 1
            if not is_batched:
                input = input.unsqueeze(batch_dim)
                if hx is not None:
                    if hx.dim() != 2:
                        raise RuntimeError(
                            f"For unbatched 2-D input, hx should also be 2-D but got {hx.dim()}-D tensor")
                    hx = hx.unsqueeze(1)
            else:
                if hx is not None and hx.dim() != 3:
                    raise RuntimeError(
                        f"For batched 3-D input, hx should also be 3-D but got {hx.dim()}-D tensor")
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None
            if hx is None:
                hx = torch.zeros(self.num_layers * num_directions,
                                 max_batch_size, self.hidden_size,
                                 dtype=input.dtype, device=input.device)
                print("input is not packed, hx is none")
            else:
                # Each batch of the hidden state should match the input sequence that
                # the user believes he/she is passing in.
                hx = self.permute_hidden(hx, sorted_indices)

        assert hx is not None
        self.check_forward_args(input, hx, batch_sizes)
        assert self.mode == 'RNN_TANH' or self.mode == 'RNN_RELU'
        if batch_sizes is None:
            print("batch_sizes is none")
            if self.mode == 'RNN_TANH':
                print("input: ", input)
                print("input shape: ",input.shape)
                result = _VF.rnn_tanh(input, hx, self._flat_weights, self.bias, self.num_layers,
                                      self.dropout, self.training, self.bidirectional,
                                      self.batch_first)
            else:
                result = _VF.rnn_relu(input, hx, self._flat_weights, self.bias, self.num_layers,
                                      self.dropout, self.training, self.bidirectional,
                                      self.batch_first)
        else:
            if self.mode == 'RNN_TANH':
                result = _VF.rnn_tanh(input, batch_sizes, hx, self._flat_weights, self.bias,
                                      self.num_layers, self.dropout, self.training,
                                      self.bidirectional)
            else:
                result = _VF.rnn_relu(input, batch_sizes, hx, self._flat_weights, self.bias,
                                      self.num_layers, self.dropout, self.training,
                                      self.bidirectional)

        output = result[0]
        hidden = result[1]

        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
            return output_packed, self.permute_hidden(hidden, unsorted_indices)

        if not is_batched:  # type: ignore[possibly-undefined]
            output = output.squeeze(batch_dim)  # type: ignore[possibly-undefined]
            hidden = hidden.squeeze(1)

        return output, self.permute_hidden(hidden, unsorted_indices)

# XXX: LSTM and GRU implementation is different from RNNBase, this is because:
# 1. we want to support nn.LSTM and nn.GRU in TorchScript and TorchScript in
#    its current state could not support the python Union Type or Any Type
# 2. TorchScript static typing does not allow a Function or Callable type in
#    Dict values, so we have to separately call _VF instead of using _rnn_impls
# 3. This is temporary only and in the transition state that we want to make it
#    on time for the release
#
# More discussion details in https://github.com/pytorch/pytorch/pull/23266
#
# TODO: remove the overriding implementations for LSTM and GRU when TorchScript
# support expressing these two modules generally.

class RNNModel(nn.Module):
    def __init__(self, hyperparams, device):
        super(RNNModel, self).__init__()
        #Set cpu/gpu device
        self.device=device
        self.to(self.device)

        # Defining some parameters
        self.hyperparams=hyperparams
        self.direction="forward"
        self.hidden_dim = hyperparams["hidden_dim"]
        self.embedding_dim = hyperparams["embedding_dim"]
        self.n_layers = hyperparams["n_rnn_layers"]
        self.output_size = hyperparams["output_size"]
        

        #Defining the layers
        #Input
        self.embedding = nn.Embedding(self.output_size, self.embedding_dim)
        # RNN Layer
        self.num_directions= 2 if self.direction=="bidirectional" else 1
        #self.hidden_dim=[self.hidden_dim, self.hidden_dim*2][self.direction=="bidirectional"]
        self.rnn = RNN(self.embedding_dim, self.hidden_dim, self.n_layers, batch_first=True, bidirectional=(self.direction == "bidirectional"))
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim*self.num_directions, self.output_size)

    def forward(self, x):

        #Send input to device (gpu/cpu)
        x.to(self.device)

        #Here you define the forward pass, calling the layers you have defined in the constructor (init function)
        batch_size = x.size(0)
        
        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        #Embed the words
        embeded = self.embedding(x)

        # Passing in the input and hidden state into the model and obtaining outputs
        #x.shape: (batch, sentence_length)
        #(num_layers * num_directions, batch, hidden_size)
        #out shape: (seq_len, batch, num_directions * hidden_size)
        print("hidden:", hidden)
        print("embeded: ", embeded)
        print("Hidden shape: ", hidden.shape)
        print("Embeded shape", embeded.shape)
        out, hidden = self.rnn(embeded, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        #out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)

        out = F.log_softmax(out, dim=2)

        return out, hidden
    
    def init_hidden(self, batch_size):
        # Generates the first hidden state (just zeros)
        # (num_layers * num_directions, batch, hidden_size)
        hidden = torch.zeros(self.n_layers*self.num_directions, batch_size, self.hidden_dim, device=self.device)
        return hidden
    
    def save_model(self, path_to_saved_models, mapper,  epochs_trained, args, additional_description=""):
        """save hyperparameters, weights, model and mappings. """

        modelfolder=join(path_to_saved_models, "model_%s_%idelay_seed_%i_epoch_%i/"%(args.language, args.delay, args.seed, epochs_trained))
        if not exists(modelfolder):
            mkdir(modelfolder)
        
        #Save hyperparameters
        json.dump(self.hyperparams, open( join(modelfolder, "hyperparams.json"), 'w' ) )
            
        #Save weights in a readable format, for further analyses
        for label, weights in self.state_dict().items():
            fname=join(modelfolder, "%s_weights.json"%label)
            if self.device.type == 'cuda':
                weights_in_mem=weights.cpu()
                npw=weights_in_mem.numpy()
            else:
                npw=weights.numpy()
            np.savetxt(fname, npw)
    
        #Save model
        torch.save(self.state_dict(), join(modelfolder, "model"))

        #Save mappings
        if mapper is not None:
            mapper.save(join(modelfolder, "w2i"))

        #Save file with command line arguments (containing hyperparams but also input file, seed, etc)
        json.dump(vars(args), open( join(modelfolder, "args.json"), 'w' ) )

        return modelfolder
