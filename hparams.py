import tensorflow as tf
from text import symbols


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = tf.contrib.training.HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=500,
        batch_size=4,
        iters_per_checkpoint=1000,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=False,
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        with_tensorboard=True,
        ignore_layers=['embedding.weight'],

        ################################
        # Data Parameters              #
        ################################
        load_mel_from_disk=False,
        training_files='metadata.txt',
        text_cleaners=['english_cleaners'],

        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        sampling_rate=22050,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,

        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(symbols),
        n_position=200,
        symbols_embedding_dim=512,
        
        ################################
        # Encoder Decoder              #
        ################################
        d_model=512,
        d_o=256,
        d_k=64,
        d_v=64,
        n_head=8,
        n_layers=6,
        d_hidden=2048,

        dropout=0.1,

        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,
        
        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=1e-4,
        lr_last=1e-5,
        lr_decay_rate=0.8,
        lr_start_decay=50000,
        lr_decay_step=25000,
        
        ################################
        # Training Parameters          #
        ################################
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        mask_padding=True,  # set model's padded outputs to padded values
        
        ################################
        # WaveGlow parameters          #
        ################################
        sigma=1.0,
        n_flows=6,
        n_group=8,
        n_early_every=4,
        n_early_size=20,
        
        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1000,
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,

        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=128,
        
        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31
    )

    if hparams_string:
        tf.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams
