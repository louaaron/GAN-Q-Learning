class Generator(object):
    """
    Interface for a generator. The generator should take in
    a state and random seed and outputs a reward distrbution
    over actions
    """

    @property
    def input_state(self):
        """
        The input state

        Returns
        -------
            A placeholder tensor: the input state's placeholder tensor
        """
        pass

    @property
    def output(self):
        """
        The outputted action distribution

        Returns
        -------
            A tensor: the output tensor
        """
        pass

    @property
    def sess(self):
        """
        The session used to create the graph

        Returns
        -------
            A session: the graph's session
        """
        pass

    @property
    def input_seed(self):
        """
        The input random seed

        Returns
        -------
            A placeholder: the input seed's placeholder tensor
        """
        pass

    @property
    def trainable_variables(self):
        """
        A list of the trainable variables in our generator

        Returns
        -------
            A list of tensors: the trainable variables in this graph 
        """
        pass

class Discriminator(object):
    """
    Interface for a discriminator. The discriminator should take in
    a state, action, and expected reward and return a probability
    value
    """

    @property
    def input_state(self):
        """
        The input state

        Returns
        -------
            A placeholder tensor: the input state's placeholder tensor
        """
        pass

    @property
    def input_action(self):
        """
        The input action

        Returns
        -------
            A placeholder tensor: the input action's placeholder tensor
        """
        pass

    @property
    def output(self):
        """
        The probability output

        Returns
        -------
            A tensor: the output's tensor
        """
        pass

    @property
    def sess(self):
        """
        The session used to create a graph

        Returns
        -------
            A session: the graph's session
        """
        pass

    @property
    def input_reward(self):
        """
        The input reward

        Returns
        -------
            A placeholder tensor: the input reward's tensor
        """
        pass

    @property
    def trainable_variables(self):
        """
        A list of the trainable variables in our generator

        Returns
        -------
            A list of tensors: the trainable variables in this graph 
        """
        pass

class Discriminator_copy(object):
    """
    Interface for copying a discriminator (used for Loss function).
    The discriminator_copy object should be initialized by a discriminator
    and a new reward placeholder. This new discriminator should share weights
    and other variables with the original dis, but should be run on the 
    new_rew_input.
    """

    def __init__(self, dis, new_rew_input):
        """
        Initializes a discriminator_copy object

        Args
        ----
            dis (Discriminator) : The discriminator to copy
            new_rew_input (tf.placeholder) : a new reward input.
        """
        pass

    @property
    def input_state(self):
        """
        The input state

        Returns
        -------
            A placeholder tensor: the input state's placeholder tensor
        """
        pass

    @property
    def input_action(self):
        """
        The input action

        Returns
        -------
            A placeholder tensor: the input action's placeholder tensor
        """
        pass

    @property
    def output(self):
        """
        The outputted action distribution

        Returns
        -------
            A tensor: the output tensor
        """
        pass

    @property
    def sess(self):
        """
        The session used to create a graph

        Returns
        -------
            A session: the graph's session
        """
        pass

    @property
    def trainable_variables(self):
        """
        A list of the trainable variables in our generator

        Returns
        -------
            A list of tensors: the trainable variables in this graph 
        """
        pass