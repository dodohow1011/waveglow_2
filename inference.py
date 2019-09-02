# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import os
from scipy.io.wavfile import write
import torch
from denoiser import Denoiser
from train import load_pretrained_taco, parse_batch 
from hparams import create_hparams
from data_utils import TextMelLoader, TextMelCollate
from torch.utils.data import DataLoader
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
def main(text_files, waveglow_path, sigma, output_dir, sampling_rate, is_fp16,
         denoiser_strength):
    hparams = create_hparams()
    Taco2 = load_pretrained_taco('tacotron2.pt', hparams)    

    testset = TextMelLoader(text_files, hparams)
    collate_fn = TextMelCollate()

    test_loader = DataLoader(testset, num_workers=0, shuffle=False,
                             sampler=None,
                             batch_size=1,
                             pin_memory=False,
                             drop_last=True, collate_fn=collate_fn)
    waveglow = torch.load(waveglow_path)['model']
    # waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.cuda().eval()
    if is_fp16:
        from apex import amp
        waveglow, _ = amp.initialize(waveglow, [], opt_level="O3")

    if denoiser_strength > 0:
        denoiser = Denoiser(waveglow).cuda()

    for i, batch in enumerate(test_loader):
        text_padded, input_lengths, mel_padded, max_len, output_lengths = parse_batch(batch)
        enc_outputs, _ = Taco2((text_padded, input_lengths, mel_padded, max_len, output_lengths))
        # mel = torch.autograd.Variable(mel.cuda())
        # mel = torch.unsqueeze(mel, 0)
        # mel = mel.half() if is_fp16 else mel
        with torch.no_grad():
            mel = waveglow.infer(enc_outputs, input_lengths, sigma=sigma)
            '''if denoiser_strength > 0:
                audio = denoiser(audio, denoiser_strength)
            audio = audio * MAX_WAV_VALUE'''
        # audio = audio.squeeze()
        # mel = mel.cpu().numpy()
        # audio = audio.astype('int16')
        print (mel)
        mel = mel.squeeze()
        print (mel.size())
        mel_path = os.path.join(
            output_dir, "{}_synthesis.pt".format(i))
        torch.save(mel, mel_path)
        print(mel_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--filelist_path", required=True)
    parser.add_argument('-w', '--waveglow_path',
                        help='Path to waveglow decoder checkpoint with model')
    parser.add_argument('-o', "--output_dir", required=True)
    parser.add_argument("-s", "--sigma", default=1.0, type=float)
    parser.add_argument("--sampling_rate", default=22050, type=int)
    parser.add_argument("--is_fp16", default=False, action="store_true")
    parser.add_argument("-d", "--denoiser_strength", default=0.0, type=float,
                        help='Removes model bias. Start with 0.1 and adjust')

    args = parser.parse_args()

    main(args.filelist_path, args.waveglow_path, args.sigma, args.output_dir,
         args.sampling_rate, args.is_fp16, args.denoiser_strength)
