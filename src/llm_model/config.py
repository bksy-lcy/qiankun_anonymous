import argparse

def get_args():
    parser = argparse.ArgumentParser(description='model')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=16, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length') # input_len/2
    parser.add_argument('--pred_len', type=int, default=4, help='prediction sequence length')
    parser.add_argument('--token_len', type=int, default=5, help='token length')
    # model define
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--patch_len', type=int, default=32, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--mlp_hidden_dim', type=int, default=1024, help='mlp hidden dim')
    parser.add_argument('--mlp_hidden_layers', type=int, default=2, help='mlp hidden layers')
    parser.add_argument('--mlp_activation', type=str, default='tanh', help='mlp activation')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=True)
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--mix_embeds', action='store_true', help='mix embeds', default=True)
    args = parser.parse_args()

    return args