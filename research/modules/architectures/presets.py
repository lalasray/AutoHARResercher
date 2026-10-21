"""Per-architecture initial configurations; see examples/architecture-provenance.json."""
from copy import deepcopy

COMMON = dict(hidden_dim=64, dropout=0., optimizer='adam', learning_rate=.001,
              optimizer_eps=1e-8, optimizer_beta1=.9, optimizer_beta2=.999,
              weight_decay=0., loss='cross_entropy', label_smoothing=0.,
              batch_size=64, max_epochs=150, patience=15, scheduler='none')
PRESETS = {
    'bilstmattention': dict(hidden_dim=64, conv_channels=[256, 128, 64, 32], kernel_size=3,
        pool_size=2, branch_channels=3, dropout=.2, head_dropout=.5,
        classifier_batch_norm=True, norm_eps=.001, batch_norm_momentum=.99,
        optimizer='nadam', optimizer_eps=1e-7, patience=150),
    'tinyhar': dict(hidden_dim=20, dropout=.1, batch_size=256, learning_rate=.0001),
    'tinierhar': dict(hidden_dim=16, conv_filters=4, depth=4, dropout=.3,
        batch_size=32, max_epochs=80, scheduler='plateau_validation_loss',
        lr_patience=7, lr_factor=.1),
    'deepconvlstm': dict(hidden_dim=128, conv_filters=64, lstm_layers=2, dropout=.5,
        optimizer='rmsprop', learning_rate=.01, optimizer_eps=1e-6, optimizer_alpha=.9,
        batch_size=100, max_epochs=300, patience=300),
    'deepconvlstmattention': dict(hidden_dim=128, conv_filters=64, lstm_layers=2, dropout=.5),
    'spectra': dict(hidden_dim=16, conv_filters=16, kernel_size=3, n_fft=32, stft_hop=16, dropout=.3),
    'hart': dict(hidden_dim=192, dropout=.3, learning_rate=.005, optimizer_eps=1e-7, max_epochs=200, patience=200, label_smoothing=.1),
    'mobilehart': dict(hidden_dim=144, dropout=.3, learning_rate=.005, optimizer_eps=1e-7, max_epochs=200, patience=200, label_smoothing=.1),
}
GENERIC = ('cnn', 'cnn_gru', 'gru', 'bilstm', 'tiny_transformer')


def model_defaults(architecture='bilstmattention'):
    if architecture not in PRESETS and architecture not in GENERIC:
        raise ValueError(f'Unknown architecture: {architecture}')
    return deepcopy(dict(COMMON, architecture=architecture, **PRESETS.get(architecture, {})))
