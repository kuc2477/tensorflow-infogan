#!/usr/bin/env python3
import os.path
import copy
import json
import argparse
import tensorflow as tf
from data import DATASETS
from model import InfoGAN
from train import train
from distributions import DISTRIBUTIONS
import utils


parser = argparse.ArgumentParser('TensorFlow implementation of InfoGAN')
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
    '--learning-rate', type=float, default=2e-5,
    help='learning rate for Adam [2e-5]'
)
parser.add_argument(
    '--beta1', type=float, default=0.5,
    help='momentum term of Adam [0.5]')
parser.add_argument(
    '--epochs', type=int, default=10,
    help='training epoch number'
)
parser.add_argument(
    '--batch-size', type=int, default=32,
    help='training batch size'
)
parser.add_argument(
    '--sample-size', type=int, default=32,
    help='generator sample size'
)
parser.add_argument(
    '--statistics-log-interval', type=int, default=30,
    help='number of batches per scalar logging'
)
parser.add_argument(
    '--image-log-interval', type=int, default=300,
    help='number of batches per image logging'
)
parser.add_argument(
    '--checkpoint-interval', type=int, default=1000,
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
    '--sample-dir', default='figures',
    help='directory of generated figures'
)
parser.add_argument(
    '--checkpoint-dir', default='checkpoints',
    help='directory of trained models'
)
parser.add_argument(
    '--log-dir', default='logs',
    help='directory of summaries'
)
parser.add_argument('--resume', action='store_true')

main_command = parser.add_mutually_exclusive_group(required=True)
main_command.add_argument(
    '--test', action='store_false', dest='train',
    help='flag defining whether it is in test mode'
)
main_command.add_argument(
    '--train', action='store_true',
    help='flag defining whether it is in train mode'
)


def _patch_dataset_specific_configs(args):
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
    raw_config = parser.parse_args()
    config = _patch_dataset_specific_configs(copy.deepcopy(raw_config))
    config = _patch_with_concrete_distributions(config)
    print(json.dumps(
        _patch_dataset_specific_configs(raw_config).__dict__,
        sort_keys=True, indent=4
    ))

    # test argument sanity
    assert config.z_sizes, 'noise latent code size must be defined'
    assert config.z_distributions, (
        'noise latent code distributions must be defined'
    )
    assert config.c_sizes, 'regularized latent code size must be defined'
    assert config.c_distributions, (
        'regularized latent code distributions must be defined'
    )
    assert len(config.z_distributions) == len(config.z_sizes), (
        'noise latent code specs(distributions and sizes) should be '
        'in same length.'
    )
    assert len(config.c_distributions) == len(config.c_sizes), (
        'regularized latent code specs(distributions and sizes) should be '
        'in same length.'
    )

    # compile the model
    infogan = InfoGAN(
        label=config.dataset,
        z_distributions=config.z_distributions,
        c_distributions=config.c_distributions,
        batch_size=config.batch_size,
        reg_rate=config.reg_rate,
        image_size=config.image_size,
        channel_size=config.channel_size,
        q_hidden_size=config.q_hidden_size,
        g_filter_number=config.g_filter_number,
        d_filter_number=config.d_filter_number,
        g_filter_size=config.g_filter_size,
        d_filter_size=config.d_filter_size,
    )

    # train / test the model
    if config.train:
        train(infogan, config)
    else:
        with tf.Session() as sess:
            name = '{}_test_figures'.format(infogan.name)
            utils.load_checkpoint(sess, infogan, config)
            utils.test_samples(sess, infogan, name, config)
            print('=> generated test figures for {} at {}'.format(
                infogan.name, os.path.join(config.sample_dir, name)
            ))


if __name__ == '__main__':
    tf.app.run()
