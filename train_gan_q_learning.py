import numpy as np
import tensorflow as tf
import utils
import gym

def learn(env,
          sess,
          episodes,
          buffer_size,
          reward_discount,
          dis,
          dis_copy,
          gen,
          learning_rate=0.0005,
          optimizer=tf.train.RMSPropOptimizer,
          n_dis=1,
          n_gen=1,
          lambda_=10,
          batch_size=64):
    """
    Code for the algorithm found in https://arxiv.org/abs/1805.04874
    GAN Q-Learning learns a probaility distrubtion for Z(s, a), the distributional
    value function (Q(s, a) is the case when the distrubtion is singular).

    Note that the algorithm described in figure 1 or the paper has some typos, which 
    are corrected here.

    Args
    ----
        env (gym.env) :
            The environment for training
        sess (int) :
            The session of both the discriminator and generator
        episodes (int) :
            The number of episodes to train the algorithm on
        buffer_size (int) :
            The size of the buffer
        reward_discount (float) :
            The amount of future reward to consider
        dis (neural_network.discriminator) :
            The architecture of the discriminator
        dis_copy (neural_network.discriminator_copy) :
            The architecture of the discriminator copier
        gen (neural_network.generator) :
            The architecure of the generator
        learning_rate (float - 0.0005) :
            The learning rate
        optimizer (tf.train.Optimizer - tf.train.RMSPropOptimizer) :
            The optimization initialization function
        n_dis (int - 1) :
            The number of discriminator updates per episode
        n_gen (int - 1) :
            The number of generator updates per episode
        lambda_ (float - 10) :
            The gradient penalty coefficient (0 for WGAN optimization)
        batch_size (int - 64) :
            The batch_size for training
    """

    #Assertion statements (make sure session remains the same across graphs)
    assert sess == dis.sess
    assert sess == gen.sess

    #Reset environment
    last_obs = env.reset()

    #The gradient for loss function
    grad_val_ph = tf.placeholder(tf.float32, shape=dis.input.get_shape())
    grad_dis = dis_copy(sess, dis, grad_val_ph)

    #The generator-discriminator for loss function
    gen_dis = dis_copy(sess, dis, tf.reduce_max(gen.output))

    #optimization 
    optim = optimizer(learning_rate=learning_rate)

    #loss functions
    dis_loss = tf.reduce_mean(tf.squeeze(
        dis - gen_dis + lambda_ * tf.square(tf.gradients(grad_dis.output, grad_val_ph)[0] - 1)
    ))
    gen_loss = tf.reduce_mean(-tf.squeeze(gen_dis))

    #buffer
    buffer = utils.ReplayBuffer(buffer_size, 1)

    #training algorithm

    #number of episodes to train
    for _ in range(episodes):
        #loop through all the steps
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

            #update discriminator n_dis times
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

                sess.run(optim.minimize(dis_loss, var_list=dis.trainable_variables), feed_dict={
                    gen.input_seed : batch_z,
                    gen.input_state : obs_batch,
                    gen_dis.input_state : obs_batch,
                    gen_dis.input_action : act_batch,
                    dis.input_reward : batch_y,
                    dis.input_state : obs_batch,
                    dis.input_action : act_batch,
                    grad_dis.input_state : obs_batch,
                    grad_dis.input_action : act_batch,
                    grad_val_ph : predict_x
                })

            #update the generator n_gen times
            for _ in range(n_gen):
                obs_batch, act_batch, _, _, _ = (buffer.sample(batch_size))
                batch_z = np.random.normal(0, 1, size=[batch_size])   
                sess.run(optim.minimize(gen_loss, var_list=gen.trainable_variables), feed_dict={
                    gen.input_seed : batch_z,
                    gen.input_state : obs_batch,
                    gen_dis.input_state : obs_batch,
                    gen_dis.input_action: act_batch
                })
