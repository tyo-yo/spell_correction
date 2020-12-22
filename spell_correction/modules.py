from typing import Dict, Optional, Tuple

import torch
from allennlp.modules import Attention
from allennlp.nn import util
from allennlp_models.generation.modules.decoder_nets.decoder_net import DecoderNet
from allennlp_models.generation.modules.decoder_nets.lstm_cell import LstmCellDecoderNet
from overrides import overrides
from torch.nn import LSTM


@DecoderNet.register("lstm")
class LstmDecoderNet(DecoderNet):
    """
    This decoder net implements simple decoding network with LSTMCell and Attention.
    # Parameters
    decoding_dim : `int`, required
        Defines dimensionality of output vectors.
    target_embedding_dim : `int`, required
        Defines dimensionality of input target embeddings.  Since this model takes it's output on a previous step
        as input of following step, this is also an input dimensionality.
    attention : `Attention`, optional (default = `None`)
        If you want to use attention to get a dynamic summary of the encoder outputs at each step
        of decoding, this is the function used to compute similarity between the decoder hidden
        state and encoder outputs.
    """

    def __init__(
        self,
        decoding_dim: int,
        target_embedding_dim: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        attention: Optional[Attention] = None,
        bidirectional_input: bool = False,
    ) -> None:

        super().__init__(
            decoding_dim=decoding_dim,
            target_embedding_dim=target_embedding_dim,
            decodes_parallel=False,
        )

        # In this particular type of decoder output of previous step passes directly to the input of current step
        # We also assume that decoder output dimensionality is equal to the encoder output dimensionality
        decoder_input_dim = self.target_embedding_dim

        # Attention mechanism applied to the encoder output for each step.
        self._attention = attention

        if self._attention:
            # If using attention, a weighted average over encoder outputs will be concatenated
            # to the previous target embedding to form the input to the decoder at each
            # time step. encoder output dim will be same as decoding_dim
            decoder_input_dim += decoding_dim

        # We'll use an LSTM cell as the recurrent cell that produces a hidden state
        # for the decoder at each time step.
        self._decoder_cell = LSTM(
            decoder_input_dim,
            self.decoding_dim,
            num_layers=num_layers,
            bias=bias,
            dropout=dropout,
        )
        self._bidirectional_input = bidirectional_input

    _prepare_attended_input = LstmCellDecoderNet._prepare_attended_input
    init_decoder_state = LstmCellDecoderNet.init_decoder_state

    @overrides
    def forward(
        self,
        previous_state: Dict[str, torch.Tensor],
        encoder_outputs: torch.Tensor,
        source_mask: torch.BoolTensor,
        previous_steps_predictions: torch.Tensor,
        previous_steps_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:

        decoder_hidden = previous_state["decoder_hidden"]
        decoder_context = previous_state["decoder_context"]

        # shape: (group_size, output_dim)
        last_predictions_embedding = previous_steps_predictions[:, -1]

        if self._attention:
            # shape: (group_size, encoder_output_dim)
            attended_input = self._prepare_attended_input(
                decoder_hidden, encoder_outputs, source_mask
            )

            # shape: (group_size, decoder_output_dim + target_embedding_dim)
            decoder_input = torch.cat((attended_input, last_predictions_embedding), -1)
        else:
            # shape: (group_size, target_embedding_dim)
            decoder_input = last_predictions_embedding

        # shape: (1, group_size, decoder_input_dim)
        decoder_input = decoder_input.unsqueeze(0)

        # shape (decoder_hidden): (batch_size, decoder_output_dim)
        # shape (decoder_context): (batch_size, decoder_output_dim)
        _, (decoder_hidden, decoder_context) = self._decoder_cell(
            decoder_input.float(), (decoder_hidden.float(), decoder_context.float())
        )

        return (
            {"decoder_hidden": decoder_hidden, "decoder_context": decoder_context},
            decoder_hidden,
        )
