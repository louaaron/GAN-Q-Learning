import numpy as np
import tensorflow as tf
import utils
import gym
import neural_network

def learn(env,
          sess,
          episodes,
          buffer_size,
          reward_discount,
          dis,
          gen,
          learning_rate=0.0000005,
          n_dis=1,
          n_gen=1,
          lambda_=10,
          batch_size=64):

    assert sess == dis.sess
    assert sess == gen.sess

    last_obs = env.reset()

    grad_val_ph = tf.placeholder(tf.float32, shape = dis.input.get_shape())
    grad_dis = neural_network.discriminator_copy(
        sess, dis, grad_val_ph
    )
    gen_dis = neural_network.discriminator_copy(sess, dis, tf.reduce_max(gen.output))

    optimizer = tf.train.RMSPropOptimizer(learning_rate)

    dis_loss = tf.reduce_mean(tf.squeeze(
        dis - gen_dis + lambda_ * tf.square(tf.gradients(grad_dis.output, grad_val_ph)[0] - 1)
    ))
    gen_loss = tf.reduce_mean(-tf.squeeze(gen_dis))

    buffer = utils.ReplayBuffer(buffer_size, 1)
    for _ in range(episodes):
        for _ in range(env._max_episode_steps):
            gen_seed = np.random.normal(0, 1)
            action_results = sess.run(gen.output, feed_dict={
                gen.input_state : np.array([last_obs]), 
                gen.input_seed : np.array([gen_seed])
            })[0]
            optimal_action = np.amax(action_results)

            next_obs, reward, done, _ = env.step(optimal_action)
            idx = buffer.store_frame(last_obs)
            buffer.store_effect(idx, optimal_action, reward, done)

            if done:
                last_obs = env.reset()
            else:
                last_obs = next_obs

            if not buffer.can_sample(batch_size):
                continue

            for _ in range(n_dis):
                obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = (
                    buffer.sample(batch_size)
                )
                batch_z = np.random.normal(0, 1, size=[batch_size])
                batch_y = []
                for i in range(batch_size):
                    if done_batch[i]:
                        batch_y.append(rew_batch[i])
                    else:
                        expected_ar = sess.run(gen.output, feed_dict={
                            gen.input_state : obs_batch[i],
                            gen.input_seed: batch_z[i]
                        })
                        future_reward = np.max(expected_ar)
                        batch_y.append(rew_batch[i] + reward_discount * future_reward)
                batch_y = np.array(batch_y)
                epsilons = np.random.uniform(0, 1, batch_size)
                predict_x = []
                for i in range(batch_size):
                    predict_x.append(epsilons[i] * batch_y[i] + (1 - epsilons[i]) * np.max(sess.run(gen.output, 
                    feed_dict={
                        gen.input_state : next_obs_batch[i],
                        gen.input_seed : batch_z[i]
                    })))
                predict_x = np.array(predict_x)

            

                sess.run(optimizer.minimize(dis_loss, var_list=dis.trainable_variables), feed_dict={
                    gen.input_seed : batch_z,
                    gen.input_state : obs_batch,
                    gen_dis.input_state : obs_batch,
                    gen_dis.input_action : act_batch,
                    dis.input_value : batch_y,
                    dis.input_state : obs_batch,
                    dis.input_action : act_batch,
                    grad_dis.input_state : obs_batch,
                    grad_dis.input_action : act_batch,
                    grad_val_ph : predict_x
                })

            for _ in range(n_gen):
                obs_batch, act_batch, _, _, _ = (buffer.sample(batch_size))
                batch_z = np.random.normal(0, 1, size=[batch_size])   
                sess.run(optimizer.minimize(gen_loss, var_list=gen.trainable_variables), feed_dict={
                    gen.input_seed : batch_z,
                    gen.input_state : obs_batch,
                    gen_dis.input_state : obs_batch,
                    gen_dis.input_action: act_batch
                })
