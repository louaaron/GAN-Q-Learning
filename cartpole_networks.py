import numpy as np
import tensorflow as tf
import neural_network as nn

class Generator(nn.Generator):
    """
    Example OpenAI-Gym Generator architecture. 
    """
    def __init__(self, sess):
        """
        Args
        ----
            sess : the tensorflow session to be used
        """
        self.sess_ = sess
        with tf.variable_scope('gen'):
            self.input_state_ = tf.placeholder(tf.float32, shape=[None, 4], name='input_state')
            self.input_seed_ = tf.placeholder(tf.float32, shape=[None, 1], name='input_seed')
            self.concat = tf.concat([self.input_state_, self.input_seed_], 1, name='concat')
            self.hidden = tf.layers.dense(self.concat, 8, activation=tf.nn.relu, name='hidden')
            self.output_ = tf.layers.dense(self.hidden, 2, name='output')
        self.sess.run(tf.global_variables_initializer())

    @property
    def input_state(self):
        """
        The input state of shape [None, 4]

        Returns
        -------
            A placeholder tensor: the input state's placeholder tensor
        """
        return self.input_state_

    @property
    def output(self):
        """
        The outputted action distribution of shape [None, 2]

        Returns
        -------
            A tensor: the output tensor
        """
        return self.output_

    @property
    def sess(self):
        """
        The session used to create the graph

        Returns
        -------
            A session: the graph's session
        """
        return self.sess_

    @property
    def input_seed(self):
        """
        The input random seed

        Returns
        -------
            A placeholder: the input seed's placeholder tensor
        """
        return self.input_seed_

    @property
    def trainable_variables(self):
        """
        A list of the trainable variables in our generator

        Returns
        -------
            A list of tensors: the trainable variables in this graph 
        """
        return tf.trainable_variables('gen')

class Discriminator(nn.Discriminator):
    """
    Example OpenAI-Gym Discriminator Architecture
    """
    def __init__(self, sess):
        """
        Args
        ----
            sess : the tensorflow session to be used
        """
        self.sess_ = sess
        with tf.variable_scope('dis'):
            self.input_state_ = tf.placeholder(tf.float32, shape=[None, 4], name='input_state')
            self.input_reward_ = tf.placeholder(tf.float32, shape=[None], name='input_reward')
            self.input_action_ = tf.placeholder(tf.float32, shape=[None, 1], name='input_action')
            self.input_reward_exp = tf.expand_dims(self.input_reward_, axis=-1, name='input_reward_expanded')
            self.concat = tf.concat([self.input_state_, self.input_reward_exp, self.input_action_], axis=1, name='concat')
            self.hidden = tf.layers.dense(self.concat, 8, activation=tf.nn.relu, name='hidden')
            self.output_ = tf.layers.dense(self.hidden, 1, activation=tf.sigmoid, name='output')
        self.sess.run(tf.global_variables_initializer())

    @property
    def input_state(self):
        """
        The input state of shape [None, 4]

        Returns
        -------
            A placeholder tensor: the input state's placeholder tensor
        """
        return self.input_state_

    @property
    def input_action(self):
        """
        The input action of shape [None, 1]

        Returns
        -------
            A placeholder tensor: the input action's placeholder tensor
        """
        return self.input_action_

    @property
    def output(self):
        """
        The probability output of shape [None, 1]

        Returns
        -------
            A tensor: the output's tensor
        """
        return self.output_

    @property
    def sess(self):
        """
        The session used to create a graph

        Returns
        -------
            A session: the graph's session
        """
        return self.sess_

    @property
    def input_reward(self):
        """
        The input reward

        Returns
        -------
            A placeholder tensor: the input reward's tensor
        """
        return self.input_reward_

    @property
    def trainable_variables(self):
        """
        A list of the trainable variables in our generator

        Returns
        -------
            A list of tensors: the trainable variables in this graph 
        """
        return tf.trainable_variables('dis')

class Discriminator_copy(nn.Discriminator_copy):
    """
    Example OpenAI-Gym Discriminator Copying method
    """
    def __init__(self, dis, new_rew_input):
        """
        Initializes a discriminator_copy object

        Args
        ----
            dis (Discriminator) : The discriminator to copy
            new_rew_input (tf.placeholder) : a new reward input.
        """
        self.sess_ = dis.sess

        #reuse the variables
        with tf.variable_scope('dis', reuse=tf.AUTO_REUSE):
            self.input_state_ = tf.placeholder(tf.float32, shape=[None, 4], name='input_state')
            self.input_reward_ = new_rew_input
            self.input_action_ = tf.placeholder(tf.float32, shape=[None, 1], name='input_action')
            self.input_reward_exp = tf.expand_dims(self.input_reward_, axis=-1, name='input_reward_expanded')
            self.concat = tf.concat([self.input_state_, self.input_reward_exp, self.input_action_], axis=1, name='concat_copy')
            self.hidden_ker = tf.get_variable('hidden/kernel')
            self.hidden_bias = tf.get_variable('hidden/bias')
            self.output_ker = tf.get_variable('output/kernel')
            self.output_bias = tf.get_variable('output/bias')

        self.hidden = tf.matmul(self.concat, self.hidden_ker) + self.hidden_bias
        self.output_ = tf.sigmoid(tf.matmul(self.hidden, self.output_ker) + self.output_bias)

    @property
    def input_state(self):
        """
        The input state of shape [None, 4]

        Returns
        -------
            A placeholder tensor: the input state's placeholder tensor
        """
        return self.input_state_

    @property
    def input_action(self):
        """
        The input action of shape [None, 1]

        Returns
        -------
            A placeholder tensor: the input action's placeholder tensor
        """
        return self.input_action_

    @property
    def output(self):
        """
        The probability output of shape [None, 1]

        Returns
        -------
            A tensor: the output's tensor
        """
        return self.output_

    @property
    def sess(self):
        """
        The session used to create a graph

        Returns
        -------
            A session: the graph's session
        """
        return self.sess_

    @property
    def input_reward(self):
        """
        The input reward

        Returns
        -------
            A placeholder tensor: the input reward's tensor
        """
        return self.input_reward_

    @property
    def trainable_variables(self):
        """
        A list of the trainable variables in our generator

        Returns
        -------
            A list of tensors: the trainable variables in this graph
        """
        return tf.trainable_variables('dis')
