import tensorflow as tf
import train_gan_q_learning as train
import cartpole_networks as networks
import gym

def main():
    sess = tf.Session()
    gen = networks.Generator(sess)
    dis = networks.Discriminator(sess)
    dis_copy = networks.Discriminator_copy

    env = gym.make('CartPole-v0')
    train.learn(env,
                sess,
                1000,
                10000, 
                0.99, 
                dis,
                dis_copy,
                gen,
                n_gen=5,
                log_dir='./logs/')

if __name__ == '__main__' : main()
