import tensorflow as tf
from tqdm import tqdm
import utils
from data import DATASETS, DATASET_LENGTH_GETTERS


def train(model, config, session=None):
    # define session if needed.
    session = session or tf.Session()

    # define summaries.
    summary_writer = tf.summary.FileWriter(config.log_dir, session.graph)
    image_summary = tf.summary.image(
        'generated images', model.g, max_outputs=8
    )
    statistics_summaries = tf.summary.merge([
        tf.summary.scalar('discriminator loss', model.d_loss),
        tf.summary.scalar(
            'estimated mutual information between c and c|g',
            model.estimated_mutual_information_between_c_and_c_given_g
        ),
        tf.summary.histogram(
            'estimated parameters of distribution c|g',
            model.estimated_parameters_of_distribution_c_given_g
        )
    ])

    # define optimizers
    d_trainer = tf.train.AdamOptimizer(
        learning_rate=config.learning_rate,
        beta1=config.beta1
    )
    g_trainer = tf.train.AdamOptimizer(
        learning_rate=config.learning_rate,
        beta1=config.beta1,
    )

    # define parameter update tasks
    d_grads = d_trainer.compute_gradients(model.d_loss, var_list=(
        model.d_vars + model.q_vars
    ))
    g_grads = g_trainer.compute_gradients(model.g_loss, var_list=(
        model.g_vars + model.q_vars
    ))
    update_d = d_trainer.apply_gradients(d_grads)
    update_g = g_trainer.apply_gradients(g_grads)

    # main training session context
    with session:
        if config.resume:
            epoch_start = (
                utils.load_checkpoint(session, model, config)
                // DATASET_LENGTH_GETTERS[config.dataset]()
            ) + 1
        else:
            epoch_start = 1
            session.run(tf.global_variables_initializer())

        for epoch in range(epoch_start, config.epochs+1):
            dataset = DATASETS[config.dataset](config.batch_size)
            dataset_length = DATASET_LENGTH_GETTERS[config.dataset]()
            dataset_stream = tqdm(enumerate(dataset, 1))

            for batch_index, xs in dataset_stream:
                # where are we?
                iteration = (epoch-1)*dataset_length + batch_index

                # run the discriminator trainer.
                zs_d, cs_d = session.run([
                    model.z_distribution.sample_prior(config.batch_size),
                    model.c_distribution.sample_prior(config.batch_size)
                ])
                _, d_loss, estimated_mutual_information = session.run([
                    update_d, model.d_loss,
                    model.estimated_mutual_information_between_c_and_c_given_g,
                ], feed_dict={
                    model.z_in: zs_d,
                    model.c_in: cs_d,
                    model.image_in: xs
                })

                # run the generator trainer.
                for _ in range(config.generator_update_ratio):
                    zs_g, cs_g = session.run([
                        model.z_distribution.sample_prior(config.batch_size),
                        model.c_distribution.sample_prior(config.batch_size)
                    ])
                    _, g_loss = session.run(
                        [update_g, model.g_loss], feed_dict={
                            model.z_in: zs_g,
                            model.c_in: cs_g
                        }
                    )

                dataset_stream.set_description((
                    'epoch: {epoch}/{epochs} | '
                    'iteration: {iteration} | '
                    'progress: [{trained}/{total}] ({progress:.0f}%) | '
                    'g loss: {g_loss:.3f} | '
                    'd loss: {d_loss:.3f} | '
                    'I(c;c|g): {estimated_mutual_information:.3f} '
                ).format(
                    epoch=epoch,
                    epochs=config.epochs,
                    iteration=iteration,
                    trained=batch_index*config.batch_size,
                    total=dataset_length,
                    progress=(
                        100.
                        * batch_index
                        * config.batch_size
                        / dataset_length
                    ),
                    g_loss=g_loss,
                    d_loss=d_loss,
                    estimated_mutual_information=estimated_mutual_information,
                ))

                # log the generated samples.
                if iteration % config.image_log_interval == 0:
                    zs, cs = session.run([
                        model.z_distribution.sample_prior(config.batch_size),
                        model.c_distribution.sample_prior(config.batch_size)
                    ])
                    summary_writer.add_summary(session.run(
                        image_summary, feed_dict={
                            model.z_in: zs,
                            model.c_in: cs,
                        }
                    ), iteration)

                # log the statistics.
                if iteration % config.statistics_log_interval == 0:
                    zs, cs = session.run([
                        model.z_distribution.sample_prior(config.batch_size),
                        model.c_distribution.sample_prior(config.batch_size)
                    ])
                    summary_writer.add_summary(session.run(
                        statistics_summaries, feed_dict={
                            model.z_in: zs,
                            model.c_in: cs,
                            model.image_in: xs
                        }
                    ), iteration)

                # save the model.
                if iteration % config.checkpoint_interval == 0:
                    utils.save_checkpoint(session, model, iteration, config)
