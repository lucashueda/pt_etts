import torch
from torch import nn

from TTS.tts.layers.gst_layers import GST
from TTS.tts.layers.tacotron2 import Decoder, Encoder, Postnet
from TTS.tts.models.tacotron_abstract import TacotronAbstract

# TODO: match function arguments with tacotron
class Tacotron2(TacotronAbstract):
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
                 agg_style_space = True,
                 use_style_lookup = False,
                 lookup_style_dim = 64,
                 use_prosodic_linear = False,
                 prosodic_dim = 64,
                 multi_speaker_agg = 'concatenate',
                 style_agg = 'concatenate'):
        super(Tacotron2,
              self).__init__(num_chars, num_speakers, num_styles, r, postnet_output_dim,
                             decoder_output_dim, attn_type, attn_win,
                             attn_norm, prenet_type, prenet_dropout,
                             forward_attn, trans_agent, forward_attn_mask,
                             location_attn, attn_K, separate_stopnet,
                             bidirectional_decoder, double_decoder_consistency,
                             ddc_r, encoder_in_features, decoder_in_features,
                             speaker_embedding_dim, gst, gst_embedding_dim,
                             gst_num_heads, gst_style_tokens, gst_use_speaker_embedding,
                             gst_use_linear_style_target, use_only_reference, lookup_speaker_dim,
                             num_prosodic_features, agg_style_space, use_style_lookup, lookup_style_dim,
                             use_prosodic_linear, prosodic_dim, multi_speaker_agg, style_agg)

        # speaker embedding layer
        if self.num_speakers > 1:
            if not self.embeddings_per_sample:
                # speaker_embedding_dim = int(gst_embedding_dim/2)
                speaker_embedding_dim = self.lookup_speaker_dim
                self.speaker_embedding = nn.Embedding(self.num_speakers, speaker_embedding_dim)
                self.speaker_embedding.weight.data.normal_(0, 0.3)

        # speaker and gst embeddings is concat in decoder input
        if ((self.num_speakers > 1)&(self.multi_speaker_agg == 'concatenate')):
            self.decoder_in_features += speaker_embedding_dim # add speaker embedding dim

        # Add style embedding look up, make sure to not use gst_use_linear_style_target if using style look up 
        if((self.num_styles > 1)&(self.use_style_lookup)&(self.style_agg == 'concatenate')):
            style_embedding_dim = self.lookup_style_dim
            self.decoder_in_features += style_embedding_dim

        if((self.num_styles > 1)&(self.use_style_lookup)):
            self.style_embedding = nn.Embedding(self.num_styles, style_embedding_dim)
            self.style_embedding.weight.data.normal_(0, 0.3)


        # If use linear prosodic info 
        if(self.use_prosodic_linear):
            self.prosodic_linear = nn.Linear(self.num_prosodic_features, self.prosodic_dim, bias=False)
            self.decoder_in_features += self.prosodic_dim
        else:
            # Add prosodic features in decoder_in_features, default is 0
            self.decoder_in_features += self.num_prosodic_features


        # embedding layer
        self.embedding = nn.Embedding(num_chars, 512, padding_idx=0)

        # base model layers
        self.encoder = Encoder(self.encoder_in_features)
        self.decoder = Decoder(self.decoder_in_features, self.decoder_output_dim, r, attn_type, attn_win,
                               attn_norm, prenet_type, prenet_dropout,
                               forward_attn, trans_agent, forward_attn_mask,
                               location_attn, attn_K, separate_stopnet)
        self.postnet = Postnet(self.postnet_output_dim)

        # global style token layers
        if self.gst:
            self.gst_layer = GST(num_mel=80,
                                 num_heads=self.gst_num_heads,
                                 num_style_tokens=self.gst_style_tokens,
                                 gst_embedding_dim=self.gst_embedding_dim,
                                 speaker_embedding_dim=speaker_embedding_dim if self.embeddings_per_sample and self.gst_use_speaker_embedding else None,
                                 use_only_reference = self.use_only_reference)
                                       
            # If enabled, we use a linear dense layer to force the embedding space to be linear 
            # separable. Note that, by our implementation, num_styles will be n-1 #styles, because 
            # we use neutral one to be the vector [0,0,0]. But if semi supervised is on it will be 
            # same len of unique style values, because then we use CrossEntropyLoss class
            if self.gst_use_linear_style_target:
                if self.agg_style_space:
                    self.linear_style_target_layer = nn.Linear(self.gst_embedding_dim + self.num_prosodic_features, self.num_styles) 
                else:
                    self.linear_style_target_layer = nn.Linear(self.gst_embedding_dim, self.num_styles) 

        # backward pass decoder
        if self.bidirectional_decoder:
            self._init_backward_decoder()
        # setup DDC
        if self.double_decoder_consistency:
            self.coarse_decoder = Decoder(
                self.decoder_in_features, self.decoder_output_dim, ddc_r, attn_type,
                attn_win, attn_norm, prenet_type, prenet_dropout, forward_attn,
                trans_agent, forward_attn_mask, location_attn, attn_K,
                separate_stopnet)

    @staticmethod
    def shape_outputs(mel_outputs, mel_outputs_postnet, alignments):
        mel_outputs = mel_outputs.transpose(1, 2)
        mel_outputs_postnet = mel_outputs_postnet.transpose(1, 2)
        return mel_outputs, mel_outputs_postnet, alignments

    def forward(self, text, text_lengths, mel_specs=None, mel_lengths=None, speaker_ids=None, \
        speaker_embeddings=None, pitch_range=None, speaking_rate=None, energy=None, style_ids = None):
        # compute mask for padding
        # B x T_in_max (boolean)
        input_mask, output_mask = self.compute_masks(text_lengths, mel_lengths)
        # B x D_embed x T_in_max
        embedded_inputs = self.embedding(text).transpose(1, 2)
        # B x T_in_max x D_en
        encoder_outputs = self.encoder(embedded_inputs, text_lengths)
        if self.gst:
            # B x gst_dim (logits are the gst logits of style tokens)
            encoder_outputs, gst_outputs, logits = self.compute_gst(encoder_outputs,
                                               mel_specs,
                                               speaker_embeddings if self.gst_use_speaker_embedding else None, 
                                               self.agg_style_space, 
                                               pitch_range, speaking_rate, energy, self.style_agg)
            if self.gst_use_linear_style_target:
                logits = self.linear_style_target_layer(gst_outputs)
        else:
            logits = None

        # prosodic features 
        if((not self.agg_style_space) & (self.num_prosodic_features > 0)):

            if(self.use_prosodic_linear):

                # Concat prosodic features 

                prosody_feats = torch.zeros((text.shape[0], self.num_prosodic_features)).to(text.device)

                i = 0
                if pitch_range is not None:
                    prosody_feats[:, i] = pitch_range
                    i += 1
                if speaking_rate is not None:
                    prosody_feats[:, i] = speaking_rate
                    i += 1
                if energy is not None:
                    prosody_feats[:, i] = energy
                    i += 1
                    
                prosodic_encoded = self.prosodic_linear(prosody_feats.unsqueeze(1))
                encoder_outputs = self._concat_speaker_embedding(encoder_outputs, prosodic_encoded)
            else:
                if pitch_range is not None:
                    encoder_outputs = self._concat_speaker_embedding(encoder_outputs, pitch_range.unsqueeze(1).unsqueeze(1))
                if speaking_rate is not None:
                    encoder_outputs = self._concat_speaker_embedding(encoder_outputs, speaking_rate.unsqueeze(1).unsqueeze(1))
                if energy is not None:
                    encoder_outputs = self._concat_speaker_embedding(encoder_outputs, energy.unsqueeze(1).unsqueeze(1))
                                            

        if self.num_speakers > 1:
            if not self.embeddings_per_sample:
                # B x 1 x speaker_embed_dim
                speaker_embeddings = self.speaker_embedding(speaker_ids)[:, None]
            else:
                # B x 1 x speaker_embed_dim
                speaker_embeddings = torch.unsqueeze(speaker_embeddings, 1)
            
            if(self.multi_speaker_agg == 'concatenate'):
                encoder_outputs = self._concat_speaker_embedding(encoder_outputs, speaker_embeddings)
            else:
                encoder_outputs = self._add_speaker_embedding(encoder_outputs, speaker_embeddings)

        if((self.num_styles > 1)&(self.use_style_lookup)):
            style_embeddings = self.style_embedding(style_ids)[:, None]

            if(self.style_agg == 'concatenate'):
                encoder_outputs = self._concat_speaker_embedding(encoder_outputs, style_embeddings)
            else:
                encoder_outputs = self._add_speaker_embedding(encoder_outputs, style_embeddings)

        encoder_outputs = encoder_outputs * input_mask.unsqueeze(2).expand_as(encoder_outputs)

        # B x mel_dim x T_out -- B x T_out//r x T_in -- B x T_out//r
        decoder_outputs, alignments, stop_tokens = self.decoder(
            encoder_outputs, mel_specs, input_mask)
        # sequence masking
        if mel_lengths is not None:
            decoder_outputs = decoder_outputs * output_mask.unsqueeze(1).expand_as(decoder_outputs)
        # B x mel_dim x T_out
        postnet_outputs = self.postnet(decoder_outputs)
        postnet_outputs = decoder_outputs + postnet_outputs
        # sequence masking
        if output_mask is not None:
            postnet_outputs = postnet_outputs * output_mask.unsqueeze(1).expand_as(postnet_outputs)
        # B x T_out x mel_dim -- B x T_out x mel_dim -- B x T_out//r x T_in
        decoder_outputs, postnet_outputs, alignments = self.shape_outputs(
            decoder_outputs, postnet_outputs, alignments)
        if self.bidirectional_decoder:
            decoder_outputs_backward, alignments_backward = self._backward_pass(mel_specs, encoder_outputs, input_mask)
            return decoder_outputs, postnet_outputs, alignments, stop_tokens, decoder_outputs_backward, alignments_backward, logits
        if self.double_decoder_consistency:
            decoder_outputs_backward, alignments_backward = self._coarse_decoder_pass(mel_specs, encoder_outputs, alignments, input_mask)
            return  decoder_outputs, postnet_outputs, alignments, stop_tokens, decoder_outputs_backward, alignments_backward, logits
        return decoder_outputs, postnet_outputs, alignments, stop_tokens, logits

    @torch.no_grad()
    def inference(self, text, speaker_ids=None, style_mel=None, speaker_embeddings=None, \
         pitch_range=None, speaking_rate=None, energy=None, style_ids = None):
        embedded_inputs = self.embedding(text).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)

        if self.gst:
            # B x gst_dim
            encoder_outputs, gst_outputs, logits = self.compute_gst(encoder_outputs,
                                               style_mel,
                                               speaker_embeddings if self.gst_use_speaker_embedding else None, 
                                               self.agg_style_space, 
                                               pitch_range, speaking_rate, energy,  self.style_agg)
            if self.gst_use_linear_style_target:
                logits = self.linear_style_target_layer(gst_outputs)
        else:
            logits = None

        # prosodic features
        if((not self.agg_style_space) & (self.num_prosodic_features > 0)):

            if(self.use_prosodic_linear):

                # Concat prosodic features 

                prosody_feats = torch.zeros((text.shape[0], self.num_prosodic_features)).to(text.device)

                i = 0
                if pitch_range is not None:
                    prosody_feats[:, i] = pitch_range
                    i += 1
                if speaking_rate is not None:
                    prosody_feats[:, i] = speaking_rate
                    i += 1
                if energy is not None:
                    prosody_feats[:, i] = energy
                    i += 1
                    
                prosodic_encoded = self.prosodic_linear(prosody_feats.unsqueeze(1))
                encoder_outputs = self._concat_speaker_embedding(encoder_outputs, prosodic_encoded)
            else:
                if pitch_range is not None:
                    encoder_outputs = self._concat_speaker_embedding(encoder_outputs, pitch_range.unsqueeze(1).unsqueeze(1))
                if speaking_rate is not None:
                    encoder_outputs = self._concat_speaker_embedding(encoder_outputs, speaking_rate.unsqueeze(1).unsqueeze(1))
                if energy is not None:
                    encoder_outputs = self._concat_speaker_embedding(encoder_outputs, energy.unsqueeze(1).unsqueeze(1))
                              

        if self.num_speakers > 1:
            if not self.embeddings_per_sample:
                speaker_embeddings = self.speaker_embedding(speaker_ids)[:, None]
            if(self.multi_speaker_agg == 'concatenate'):
                encoder_outputs = self._concat_speaker_embedding(encoder_outputs, speaker_embeddings)
            else:
                encoder_outputs = self._add_speaker_embedding(encoder_outputs, speaker_embeddings)

        if((self.num_styles > 1)&(self.use_style_lookup)):
            style_embeddings = self.style_embedding(style_ids)[:, None]

            if(self.style_agg == 'concatenate'):
                encoder_outputs = self._concat_speaker_embedding(encoder_outputs, style_embeddings)
            else:
                encoder_outputs = self._add_speaker_embedding(encoder_outputs, style_embeddings)

        decoder_outputs, alignments, stop_tokens = self.decoder.inference(
            encoder_outputs)
        postnet_outputs = self.postnet(decoder_outputs)
        postnet_outputs = decoder_outputs + postnet_outputs
        decoder_outputs, postnet_outputs, alignments = self.shape_outputs(
            decoder_outputs, postnet_outputs, alignments)
        return decoder_outputs, postnet_outputs, alignments, stop_tokens, logits

    def inference_truncated(self, text, speaker_ids=None, style_mel=None, speaker_embeddings=None, \
         pitch_range=None, speaking_rate=None, energy=None, style_ids = None):
        """
        Preserve model states for continuous inference
        """
        embedded_inputs = self.embedding(text).transpose(1, 2)
        encoder_outputs = self.encoder.inference_truncated(embedded_inputs)

        if self.gst:
            # B x gst_dim
            encoder_outputs, gst_outputs, logits = self.compute_gst(encoder_outputs,
                                               style_mel,
                                               speaker_embeddings if self.gst_use_speaker_embedding else None, 
                                               self.agg_style_space, 
                                               pitch_range, speaking_rate, energy,  self.style_agg)
            if self.gst_use_linear_style_target:
                logits = self.linear_style_target_layer(gst_outputs)
        else:
            logits = None

        # prosodic features 
        if((not self.agg_style_space) & (self.num_prosodic_features > 0)):

            if(self.use_prosodic_linear):

                # Concat prosodic features 

                prosody_feats = torch.zeros((text.shape[0], self.num_prosodic_features)).to(text.device)

                i = 0
                if pitch_range is not None:
                    prosody_feats[:, i] = pitch_range
                    i += 1
                if speaking_rate is not None:
                    prosody_feats[:, i] = speaking_rate
                    i += 1
                if energy is not None:
                    prosody_feats[:, i] = energy
                    i += 1
                    
                prosodic_encoded = self.prosodic_linear(prosody_feats.unsqueeze(1))
                encoder_outputs = self._concat_speaker_embedding(encoder_outputs, prosodic_encoded)
            else:
                if pitch_range is not None:
                    encoder_outputs = self._concat_speaker_embedding(encoder_outputs, pitch_range.unsqueeze(1).unsqueeze(1))
                if speaking_rate is not None:
                    encoder_outputs = self._concat_speaker_embedding(encoder_outputs, speaking_rate.unsqueeze(1).unsqueeze(1))
                if energy is not None:
                    encoder_outputs = self._concat_speaker_embedding(encoder_outputs, energy.unsqueeze(1).unsqueeze(1))
                                
            
        if self.num_speakers > 1:
            if not self.embeddings_per_sample:
                speaker_embeddings = self.speaker_embedding(speaker_ids)[:, None]
            if(self.multi_speaker_agg == 'concatenate'):
                encoder_outputs = self._concat_speaker_embedding(encoder_outputs, speaker_embeddings)
            else:
                encoder_outputs = self._add_speaker_embedding(encoder_outputs, speaker_embeddings)
                
        if((self.num_styles > 1)&(self.use_style_lookup)):
            style_embeddings = self.style_embedding(style_ids)[:, None]

            if(self.style_agg == 'concatenate'):
                encoder_outputs = self._concat_speaker_embedding(encoder_outputs, style_embeddings)
            else:
                encoder_outputs = self._add_speaker_embedding(encoder_outputs, style_embeddings)

        mel_outputs, alignments, stop_tokens = self.decoder.inference_truncated(
            encoder_outputs)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        mel_outputs, mel_outputs_postnet, alignments = self.shape_outputs(
            mel_outputs, mel_outputs_postnet, alignments)
        return mel_outputs, mel_outputs_postnet, alignments, stop_tokens, logits
