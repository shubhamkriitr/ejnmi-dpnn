# !/usr/bin/env python3
#  -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 15:21:46 2017

@author: shubham
"""
import tensorflow as tf


class Net:

    def __init__(self, arg_dict):
        # Default values
        self.verbose = True
        if "verbose" in arg_dict:
            self.verbose = arg_dict["verbose"]
        self.scope = "Net"
        self.num_of_inputs = None
        self.data_format = "channels_last"
        self.dtype = tf.float32
        self.extra_outputs = {}
        self.input_shape = None
        self.output_shape = None
        self.load_pretrained = False
        self.saved_model_path = None
        self.trainable = True
        self.saver = None  # Be sure to pass a list of vars intended to be saved
        #  and not all f them.(To avoid inadverent initialization of vars while
        #  restoring)
        self.inputs = []
        self.outputs = []
        self.costs = []#A list of dictionary e.g.{"MAE":<output tensor>}, OR just<output_tensor>
        self.true_outputs = []
        self.histogram_list = []  # list of weights to be recorded in summary
        self._built_flag = False
        self.graph = None

    def _init_all (self, arg_dict):
        """"Initializes the variables with corresponding values given in the
        dictionary arg_dict.
        """
        for key in self.__dict__:
            if key in arg_dict:
                self.__dict__[key] = arg_dict[key]
                self.print(key,"<==", arg_dict[key])
            else:
                self.print(key,"<--", self.__dict__[key])

    def _build_network(self):
        """Creates the model-graph."""
        # Override it to define network building procedure
        pass

    def build_network(self):
        """Should be explicitly called in a session to initialize the graph weig-
        hts or load it from the disk
        """
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._build_network()
            self.saver = tf.train.Saver()
        self._built_flag = True
        self.num_of_inputs = len(self.inputs)


    def load_pretrained_network (self,sess):
        if self.saver is None:
            raise AssertionError("saver has not been created.")
        if (self.load_pretrained):
            if not self._built_flag:
                self.build_network(sess)
            with sess.as_default():
                self.saver.restore(tf.get_default_session(),
                                   self.old_model_path)
                self.print("Checkpoint at:", self.old_model_path, "has been loaded.")
        else:
            raise AssertionError("Loading network has not been allowed. Set \
                                 load_pretrained flag to True to allow \
                                 loading.")

    def print(self,*args):
        if self.verbose:
            print(*args)



if __name__ == '__main__':
    net = Net({"saved_model_path":"Sk/model.h5"})
