import random
import torch
from tensorboardX import SummaryWriter
from plotting_utils import plot_alignment_to_numpy


class waveglowLogger(SummaryWriter):
    def __init__(self, logdir):
        super(waveglowLogger, self).__init__(logdir)

    def log_training(self, reduced_loss, grad_norm, learning_rate,
                     iteration):
            self.add_scalar("training.loss", reduced_loss, iteration)
            self.add_scalar("grad.norm", grad_norm, iteration)
            self.add_scalar("learning.rate", learning_rate, iteration)

    def log_alignment(self, model, dec_enc_attn, iteration):

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, dec_enc_attn[0].size(0) - 1)
        '''self.add_image(
            "encoder_self_alignment",
            plot_alignment_to_numpy(enc_slf_attn[idx].data.cpu().numpy().T),
            iteration)'''
        self.add_image(
            "decoder_encoder_alignment_1",
            plot_alignment_to_numpy(dec_enc_attn[0][idx].data.cpu().numpy().T),
            iteration)
        self.add_image(
            "decoder_encoder_alignment_2",
            plot_alignment_to_numpy(dec_enc_attn[1][idx].data.cpu().numpy().T),
            iteration)
        self.add_image(
            "decoder_encoder_alignment_3",
            plot_alignment_to_numpy(dec_enc_attn[2][idx].data.cpu().numpy().T),
            iteration)
        self.add_image(
            "decoder_encoder_alignment_4",
            plot_alignment_to_numpy(dec_enc_attn[3][idx].data.cpu().numpy().T),
            iteration)
        self.add_image(
            "decoder_encoder_alignment_5",
            plot_alignment_to_numpy(dec_enc_attn[4][idx].data.cpu().numpy().T),
            iteration)
        self.add_image(
            "decoder_encoder_alignment_6",
            plot_alignment_to_numpy(dec_enc_attn[5][idx].data.cpu().numpy().T),
            iteration)
