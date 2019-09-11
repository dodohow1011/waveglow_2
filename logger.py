import random
import torch
from tensorboardX import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy


class waveglowLogger(SummaryWriter):
    def __init__(self, logdir):
        super(waveglowLogger, self).__init__(logdir)

    def log_training(self, reduced_loss, grad_norm, learning_rate,
                     iteration):
            self.add_scalar("training.loss", reduced_loss, iteration)
            self.add_scalar("grad.norm", grad_norm, iteration)
            self.add_scalar("learning.rate", learning_rate, iteration)

    def log_alignment(self, model, dec_enc_attn, alignment, mel_padded, mel_predict, test_attn, iteration):

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        
        idx = random.randint(0, dec_enc_attn[0].size(0) - 1)
        mel_padded = mel_padded.permute(0, 2, 1)
        mel_predict = mel_predict.permute(0, 2, 1)
        '''self.add_image(
            "encoder_self_alignment",
            plot_alignment_to_numpy(enc_slf_attn[idx].data.cpu().numpy().T),
            iteration)'''
        for i in range(len(dec_enc_attn)):
            self.add_image(
                "decoder_encoder_alignment_{}".format(i),
                plot_alignment_to_numpy(dec_enc_attn[i][idx].data.cpu().numpy().T),
                iteration)
            self.add_image(
                "test_alignment_{}".format(i),
                plot_alignment_to_numpy(test_attn[len(test_attn)-i-1][idx].data.cpu().numpy().T),
                iteration)
        self.add_image(
            "target_alignment",
            plot_alignment_to_numpy(alignment[idx].data.cpu().numpy().T),
            iteration)
        self.add_image(
            "target_mel",
            plot_spectrogram_to_numpy(mel_padded[idx].data.cpu().numpy().T),
            iteration)
        self.add_image(
            "predict_mel",
            plot_spectrogram_to_numpy(mel_predict[idx].data.cpu().numpy().T),
            iteration)

