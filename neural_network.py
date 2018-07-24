import tensorflow as tf
import numpy as np 

class generator(object):
    @property
    def input_state(self):
        pass

    @property
    def output(self):
        pass

    @property
    def sess(self):
        pass

    @property
    def input_seed(self):
        pass

    @property
    def trainable_variables(self):
        pass

class discriminator(object):
    @property
    def input_state(self):
        pass

    @property
    def input_action(self):
        pass

    @property
    def output(self):
        pass

    @property
    def sess(self):
        pass

    @property
    def input_value(self):
        pass

    @property
    def trainable_variables(self):
        pass

class discriminator_copy(object):

    def __init__(self, sess, dis, new_val_input):
        pass

    @property
    def input_state(self):
        pass

    @property
    def input_action(self):
        pass

    @property
    def output(self):
        pass

    @property
    def sess(self):
        pass

    @property
    def input_value(self):
        pass

    @property
    def trainable_variables(self):
        pass