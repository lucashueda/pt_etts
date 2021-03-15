import copy
from abc import ABC, abstractmethod

import torch
from torch import nn

from TTS.tts.utils.generic_utils import sequence_mask


class TacotronAbstract(ABC, nn.Module):
    def __init__(self,
                 num_chars,
                 num_speakers,
                 num_styles,
                 r,
                 postnet_output_dim=80,
                 decoder_output_dim=80,
                 attn_type='original',
                 attn_win=False,
                 attn_norm="softmax",
                 prenet_type="original",
                 prenet_dropout=True,
                 forward_attn=False,
                 trans_agent=False,
                 forward_attn_mask=False,
                 location_attn=True,
                 attn_K=5,
                 separate_stopnet=True,
                 bidirectional_decoder=False,
                 double_decoder_consistency=False,
                 ddc_r=None,
                 encoder_in_features=512,
                 decoder_in_features=512,
                 speaker_embedding_dim=None,
                 gst=False,
                 gst_embedding_dim=512,
                 gst_num_heads=4,
                 gst_style_tokens=10,
                 gst_use_speaker_embedding=False,
                 gst_use_linear_style_target = False,
                 use_only_reference = False,
                 lookup_speaker_dim = 512,
                 num_prosodic_features = 0,
                 agg_style_space = True):
        """ Abstract Tacotron class """
        super().__init__()
        self.num_chars = num_chars
        self.r = r
        self.decoder_output_dim = decoder_output_dim
        self.postnet_output_dim = postnet_output_dim
        self.gst = gst
        self.gst_embedding_dim = gst_embedding_dim
        self.gst_num_heads = gst_num_heads
        self.gst_style_tokens = gst_style_tokens
        self.gst_use_speaker_embedding = gst_use_speaker_embedding
        self.num_speakers = num_speakers
        self.num_styles = num_styles 
        self.bidirectional_decoder = bidirectional_decoder
        self.double_decoder_consistency = double_decoder_consistency
        self.ddc_r = ddc_r
        self.attn_type = attn_type
        self.attn_win = attn_win
        self.attn_norm = attn_norm
        self.prenet_type = prenet_type
        self.prenet_dropout = prenet_dropout
        self.forward_attn = forward_attn
        self.trans_agent = trans_agent
        self.forward_attn_mask = forward_attn_mask
        self.location_attn = location_attn
        self.attn_K = attn_K
        self.separate_stopnet = separate_stopnet
        self.encoder_in_features = encoder_in_features
        self.decoder_in_features = decoder_in_features
        self.speaker_embedding_dim = speaker_embedding_dim
        self.gst_use_linear_style_target = gst_use_linear_style_target
        self.use_only_reference = use_only_reference
        self.lookup_speaker_dim = lookup_speaker_dim
        self.num_prosodic_features = num_prosodic_features
        self.agg_style_space = agg_style_space

        # layers
        self.embedding = None
        self.encoder = None
        self.decoder = None
        self.postnet = None

        # multispeaker
        if self.speaker_embedding_dim is None:
            # if speaker_embedding_dim is None we need use the nn.Embedding, with default speaker_embedding_dim
            self.embeddings_per_sample = False
        else:
            # if speaker_embedding_dim is not None we need use speaker embedding per sample
            self.embeddings_per_sample = True

        # global style token
        if self.gst:
            self.decoder_in_features += gst_embedding_dim # add gst embedding dim
            self.gst_layer = None

        # model states
        self.speaker_embeddings = None
        self.speaker_embeddings_projected = None

        # additional layers
        self.decoder_backward = None
        self.coarse_decoder = None

    #############################
    # INIT FUNCTIONS
    #############################

    def _init_states(self):
        self.speaker_embeddings = None
        self.speaker_embeddings_projected = None

    def _init_backward_decoder(self):
        self.decoder_backward = copy.deepcopy(self.decoder)

    def _init_coarse_decoder(self):
        self.coarse_decoder = copy.deepcopy(self.decoder)
        self.coarse_decoder.r_init = self.ddc_r
        self.coarse_decoder.set_r(self.ddc_r)

    #############################
    # CORE FUNCTIONS
    #############################

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def inference(self):
        pass

    #############################
    # COMMON COMPUTE FUNCTIONS
    #############################

    def compute_masks(self, text_lengths, mel_lengths):
        """Compute masks  against sequence paddings."""
        # B x T_in_max (boolean)
        device = text_lengths.device
        input_mask = sequence_mask(text_lengths).to(device)
        output_mask = None
        if mel_lengths is not None:
            max_len = mel_lengths.max()
            r = self.decoder.r
            max_len = max_len + (r - (max_len % r)) if max_len % r > 0 else max_len
            output_mask = sequence_mask(mel_lengths, max_len=max_len).to(device)
        return input_mask, output_mask

    def _backward_pass(self, mel_specs, encoder_outputs, mask):
        """ Run backwards decoder """
        decoder_outputs_b, alignments_b, _ = self.decoder_backward(
            encoder_outputs, torch.flip(mel_specs, dims=(1,)), mask,
            self.speaker_embeddings_projected)
        decoder_outputs_b = decoder_outputs_b.transpose(1, 2).contiguous()
        return decoder_outputs_b, alignments_b

    def _coarse_decoder_pass(self, mel_specs, encoder_outputs, alignments,
                             input_mask):
        """ Double Decoder Consistency """
        T = mel_specs.shape[1]
        if T % self.coarse_decoder.r > 0:
            padding_size = self.coarse_decoder.r - (T % self.coarse_decoder.r)
            mel_specs = torch.nn.functional.pad(mel_specs,
                                                (0, 0, 0, padding_size, 0, 0))
        decoder_outputs_backward, alignments_backward, _ = self.coarse_decoder(
            encoder_outputs.detach(), mel_specs, input_mask)
        # scale_factor = self.decoder.r_init / self.decoder.r
        alignments_backward = torch.nn.functional.interpolate(
            alignments_backward.transpose(1, 2),
            size=alignments.shape[1],
            mode='nearest').transpose(1, 2)
        decoder_outputs_backward = decoder_outputs_backward.transpose(1, 2)
        decoder_outputs_backward = decoder_outputs_backward[:, :T, :]
        return decoder_outputs_backward, alignments_backward

    #############################
    # EMBEDDING FUNCTIONS
    #############################

    def compute_speaker_embedding(self, speaker_ids):
        """ Compute speaker embedding vectors """
        if hasattr(self, "speaker_embedding") and speaker_ids is None:
            raise RuntimeError(
                " [!] Model has speaker embedding layer but speaker_id is not provided"
            )
        if hasattr(self, "speaker_embedding") and speaker_ids is not None:
            self.speaker_embeddings = self.speaker_embedding(speaker_ids).unsqueeze(1)
        if hasattr(self, "speaker_project_mel") and speaker_ids is not None:
            self.speaker_embeddings_projected = self.speaker_project_mel(
                self.speaker_embeddings).squeeze(1)

    def compute_gst(self, inputs, style_input, speaker_embedding=None, agg_style_space = False, \
        pitch_range = None, speaking_rate = None , energy = None):
        """ Compute global style token """
        device = inputs.device

        # print('computing gst')

        # print(inputs, style_input)

        if isinstance(style_input, dict):
            query = torch.zeros(1, 1, self.gst_embedding_dim//2).to(device)
            if speaker_embedding is not None:
                query = torch.cat([query, speaker_embedding.reshape(1, 1, -1)], dim=-1)

            _GST = torch.tanh(self.gst_layer.style_token_layer.style_tokens)
            gst_outputs = torch.zeros(1, 1, self.gst_embedding_dim).to(device)
            for k_token, v_amplifier in style_input.items():
                # print(k_token, v_amplifier)
                key = _GST[int(k_token)].unsqueeze(0).expand(1, -1, -1)
                # print(query.shape, key.shape)
                gst_outputs_att, logits = self.gst_layer.style_token_layer.attention(query, key)
                # print(gst_outputs_att.shape)
                gst_outputs = gst_outputs + gst_outputs_att * v_amplifier
        elif style_input is None:
            # print('entered here')
            logits = None
            gst_outputs = torch.zeros(1, 1, self.gst_embedding_dim).to(device)
        elif style_input.shape[2] == self.gst_embedding_dim:
            # print('entered correctly')
            logits = None
            gst_outputs = style_input
            print(gst_outputs, style_input)
        else:
            gst_outputs, logits = self.gst_layer(style_input, speaker_embedding) # pylint: disable=not-callable
        
        # Adding prosodic features to the space if agg_style_space is True
        if agg_style_space:

            if pitch_range is not None:
                gst_outputs = self._concat_speaker_embedding(gst_outputs, pitch_range.unsqueeze(1))
            if speaking_rate is not None:
                gst_outputs = self._concat_speaker_embedding(gst_outputs, speaking_rate.unsqueeze(1))
            if energy is not None:
                gst_outputs = self._concat_speaker_embedding(gst_outputs, energy.unsqueeze(1))
                     

        inputs = self._concat_speaker_embedding(inputs, gst_outputs)
        return inputs, gst_outputs, logits

    @staticmethod
    def _add_speaker_embedding(outputs, speaker_embeddings):
        speaker_embeddings_ = speaker_embeddings.expand(
            outputs.size(0), outputs.size(1), -1)
        outputs = outputs + speaker_embeddings_
        return outputs

    @staticmethod
    def _concat_speaker_embedding(outputs, speaker_embeddings):
        speaker_embeddings_ = speaker_embeddings.expand(
            outputs.size(0), outputs.size(1), -1)
        outputs = torch.cat([outputs, speaker_embeddings_], dim=-1)
        return outputs
