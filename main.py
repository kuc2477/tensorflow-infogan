import copy
import json
import argparse
import tensorflow as tf
from data import DATASETS
from model import InfoGAN
from train import train
from distributions import DISTRIBUTIONS


parser = argparse.ArgumentParser('InfoGAN CLI')
parser.add_argument(
    '--z-size', type=int, dest='z_sizes',
    action='append', nargs='+',
    help='size of noise latent code z [100]'
)
parser.add_argument(
    '--z-dist', dest='z_distributions', choices=DISTRIBUTIONS.keys(),
    action='append', nargs='+',
    help='distribution of noise latent code z'
)
parser.add_argument(
    '--c-size', type=int, dest='c_sizes',
    action='append', nargs='+',
    help='size of regularized latent code c'
)
parser.add_argument(
    '--c-dist', dest='c_distributions', choices=DISTRIBUTIONS.keys(),
    action='append', nargs='+',
    help='distribution of regularized latent code c'
)
parser.add_argument(
    '--reg-rate', type=float, default=0.5,
    nargs='?', help='information regularization coefficient'
)
parser.add_argument(
    '--image-size', type=int, default=32,
    help='size of image [32]'
)
parser.add_argument(
    '--channel-size', type=int, default=1,
    help='size of channel [1]'
)
parser.add_argument(
    '--g-filter-number', type=int, default=64,
    help='number of generator\'s filters at the last transposed conv layer'
)
parser.add_argument(
    '--d-filter-number', type=int, default=64,
    help='number of discriminator\'s filters at the first conv layer'
)
parser.add_argument(
    '--g-filter-size', type=int, default=5,
    help='generator\'s filter size'
)
parser.add_argument(
    '--d-filter-size', type=int, default=4,
    help='discriminator\'s filter size'
)
parser.add_argument(
    '--q-hidden-size', type=int, default=128,
    help='posterior latent code approximator network\'s hidden layer size'
)
parser.add_argument(
    '--learning-rate', type=float, default=0.00002,
    help='learning rate for Adam [0.00002]'
)
parser.add_argument(
    '--beta1', type=float, default=0.5,
    help='momentum term of Adam [0.5]')
parser.add_argument(
    '--dataset', default='mnist',
    help='dataset to use {}'.format(DATASETS.keys())
)
parser.add_argument(
    '--resize', action='store_true',
    help='whether to resize images on the fly or not'
)
parser.add_argument(
    '--crop', action='store_false',
    help='whether to use crop for image resizing or not'
)
parser.add_argument(
    '--iterations', type=int, default=5000,
    help='training iteration number'
)
parser.add_argument(
    '--batch-size', type=int, default=64,
    help='training batch size'
)
parser.add_argument(
    '--sample-size', type=int, default=36,
    help='generator sample size'
)
parser.add_argument(
    '--log-for-every', type=int, default=10,
    help='number of batches per logging'
)
parser.add_argument(
    '--save-for-every', type=int, default=1000,
    help='number of batches per saving the model'
)
parser.add_argument(
    '--generator-update-ratio', type=int, default=2,
    help=(
        'number of updates for generator parameters per '
        'discriminator\'s updates'
    )
)
parser.add_argument(
    '--test', action='store_true',
    help='flag defining whether it is in test mode'
)
parser.add_argument(
    '--sample-dir', default='figures',
    help='directory of generated figures'
)
parser.add_argument(
    '--model-dir', default='checkpoints',
    help='directory of trained models'
)


def _patch_dataset_specific_options(args):
    dataset_config = DATASETS[args.dataset]

    # patch dataset specific image & channel_size
    args.image_size = dataset_config.image_size or args.image_size
    args.channel_size = dataset_config.channel_size or args.channel_size

    # patch dataset specific noise latent code distribution spec
    args.z_sizes = args.z_sizes or dataset_config.z_sizes
    args.z_distributions = (
        args.z_distributions or
        dataset_config.z_distributions
    )

    # patch dataset specific regularized latent code distribution spec
    args.c_sizes = args.c_sizes or dataset_config.c_sizes
    args.c_distributions = (
        args.c_distributions or
        dataset_config.c_distributions
    )

    return args


def _patch_with_concrete_distributions(args):
    args.z_distributions = [
        DISTRIBUTIONS[name](size) for name, size in
        zip(args.z_distributions, args.z_sizes)
    ]
    args.c_distributions = [
        DISTRIBUTIONS[name](size) for name, size in
        zip(args.c_distributions, args.c_sizes)
    ]
    return args


def main(_):
    # patch and display flags with dataset's width and height
    raw_options = parser.parse_args()
    options = _patch_dataset_specific_options(copy.deepcopy(raw_options))
    options = _patch_with_concrete_distributions(options)
    print(json.dumps(
        _patch_dataset_specific_options(raw_options).__dict__,
        sort_keys=True, indent=4
    ))

    # test argument sanity
    assert options.z_sizes, 'noise latent code size must be defined'
    assert options.z_distributions, (
        'noise latent code distributions must be defined'
    )
    assert options.c_sizes, 'regularized latent code size must be defined'
    assert options.c_distributions, (
        'regularized latent code distributions must be defined'
    )
    assert len(options.z_distributions) == len(options.z_sizes), (
        'noise latent code specs(distributions and sizes) should be '
        'in same length.'
    )
    assert len(options.c_distributions) == len(options.c_sizes), (
        'regularized latent code specs(distributions and sizes) should be '
        'in same length.'
    )

    # compile the model
    dcgan = InfoGAN(
        z_distributions=options.z_distributions,
        c_distributions=options.c_distributions,
        batch_size=options.batch_size,
        reg_rate=options.reg_rate,
        image_size=options.image_size,
        channel_size=options.channel_size,
        q_hidden_size=options.q_hidden_size,
        g_filter_number=options.g_filter_number,
        d_filter_number=options.d_filter_number,
        g_filter_size=options.g_filter_size,
        d_filter_size=options.d_filter_size,
    )

    # prepare the tensorflow session
    session = tf.Session()

    # test / train the model
    if options.test:
        # TODO: NOT IMPLEMENTED YET
        print('TEST MODE NOT IMPLEMENTED YET')
    else:
        train(session, dcgan, options)


if __name__ == '__main__':
    tf.app.run()
