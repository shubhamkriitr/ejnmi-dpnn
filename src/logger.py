#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 13:26:45 2017

@author: shubham
"""
import os
from data import DataGenerator as dgen
import tensorflow as tf


def get_summary_ops_list(node_dict={}):
    """Returns a list of summary protocol buffers.(ONLY SCALARS)
        Args:
            node_dict: A dictionary of tf graph nodes e.g
            {"MAE": some_node, "DICE_LOSS": some_other_node}
        Returns:
            A list of summary operations.
    """
    op_list = []
    for node in node_dict.keys():
        op_list.append(tf.summary.scalar(name=node, tensor=node_dict[node]))
    return op_list

def get_summary_op(node_dict={}, ops_list=[], name="SUMMARY"):
    """Creates summary protocol buffers for each element in the node_dict,
    and then merges all of them to form a single ..
    Args:
        node_dict: A dictionary tf graph nodes e.g
        {"MAE": some_node, "DICE_LOSS": some_other_node}
        name: name to be assigned to the output.
        ops_list: A list of summary protocol buffers to be included in the
        final merging.
    Returns:
        A scalar `Tensor` of type `string`. The serialized `Summary` protocol
        buffer resulting from the merging.
    """
    ops_list.append[get_summary_ops_list(node_dict)]
    return tf.summary.merge(ops_list, name=name)


class LoggerBase:

    def __init__(self):
        """"""
        self.scope = "LOGS"
        self.model = None
        self.session = None
        self.log_root_folder = None  # Logs will be stored in ../train and
                                     # ../val folders.
        self.bsz = None # batch_size used during calculation of the metrics
        self.train_gen = None  # A DataGenerator object that extracts training data                               # It would be used for fetching training data
                               # without shuffling. (see data.DataGenerator's
                               # get_next_test_batch for generator description.)
        self.val_gen = None  # A  DataGenerator that extracts validation data w/o
                             # shuffling.
        self.ops = {}  # Dictionary of operations and list of operations like
                        # "train_summary_op","val_
                        # summary_op" "train_summary_reset", "train_summary_accumulate"
                        # etc. that will be created as needed.
        self.objs = {}  # Dictionary of various class instances used by the Lo-
                        # gger for maintaing logs.

    def _init_all (self, arg_dict):
        """"Initializes the variables with corresponding values given in the
        dictionary arg_dict.
        """
        for key in self.__dict__:
            if key in arg_dict:
                self.__dict__[key] = arg_dict[key]
                self.print(key,"<==", self.__dict__[key])
            else:
                self.print(key,"<--", self.__dict__[key])

    def create_summary_ops(self, node_dict={},batch_ind_node_dict={}, additional_ops=[]):
        """Adds "train_summary_op" and "val_summary_op" in the dictionary
        self.ops. It also adds "train_writer" and "val_writer" in self.objs
        Args:
            node_dict: A dictionary of graph nodes(whose value need to be
            averaged out) to be recorded in summary.
            batch_ind_node_dict:A dictionary of graph nodes(whose value
            do not change for batches over an epoch, e.g. lr) to be recorded
            in summary.
            additional_ops: (optional) Serialized summary protocol buffers that
            need to be added in the list of train summary operations.
        """
        with tf.variable_scope(self.scope):
            self._add_summary_nodes_to_graph(node_dict, batch_ind_node_dict,
                                             additional_ops)
            self._merge_summary_ops()
            self.objs["train_writer"] = tf.summary.FileWriter(self.log_root_folder
                                                              + os.sep + "train"
                                                              )
            self.objs["val_writer"] = tf.summary.FileWriter(self.log_root_folder
                                                            + os.sep + "val"
                                                            )
            self.objs["train_writer"].add_graph(self.model.graph)
            self.objs["val_writer"].add_graph(self.model.graph)
            #TODO_  Add graphto the writers after setting up everything

    def _add_summary_nodes_to_graph(self, node_dict={},batch_ind_node_dict={},
                                    additional_ops=[]):
        """Adds nodes for every metric and corresponding
        reset, accumulate, and calculate_average operation.
        """
        reset_ops = []  # list of ops to reset the variables that will store
                        # the cumulative value of the nodes in the node_dict.
        accumulate_ops =  []  # list of ops to add current value of the nodes to
                              # the variables holding corresponding cumm=ulative values.
        calculate_avg_ops = []  # list of ops to add current value of the nodes to
                                # the variables holding corresponding cumm=ulative values.
        summary_ops = []
        zero_tensor = tf.constant(0.0, dtype=tf.float32)
        one_tensor = tf.constant(1.0, dtype=tf.float32)
        sample_count = tf.get_variable("sample_count", shape=[], dtype=tf.float32)
        current_sample_count = tf.placeholder(dtype=tf.float32, shape=[],
                                             name="current_sample_count")
        reset_ops.append(sample_count.assign(zero_tensor))
        accumulate_ops.append(sample_count.assign_add(current_sample_count))

        for key in node_dict.keys():
            var = tf.get_variable(name=key, shape=[], dtype=tf.float32)
            reset_ops.append(var.assign(zero_tensor))
            accumulate_ops.append(var.assign_add(node_dict[key]*current_sample_count))
            summary_ops.append(tf.summary.scalar(name=key, tensor = var/sample_count))

        for key in batch_ind_node_dict.keys():
            tnsr = batch_ind_node_dict[key]
            summary_ops.append(tf.summary.scalar(name=key, tensor = tnsr))

        self.objs["current_sample_count"] = current_sample_count
        self.ops["reset"] = reset_ops
        self.ops["accumulate"] = accumulate_ops
        self.ops["val_summary_op_list"] = summary_ops.copy()
        for ops in additional_ops:
            summary_ops.append(ops)
        self.ops["train_summary_op_list"] = summary_ops

    def _merge_summary_ops(self):
        train_list = self.ops["train_summary_op_list"]
        val_list = self.ops["val_summary_op_list"]
        self.ops["merged_train_summaries"] = tf.summary.merge(inputs=train_list,
                                                              name = "TRAIN_SUMMARY"
                                                              )
        self.ops["merged_val_summaries"] = tf.summary.merge(inputs=val_list,
                                                            name="VAL_SUMMARY"
                                                            )

    def set_session(self,session):
        self.session = session

    def record_train_summary(self):
        """Runs the reset, accumulate and merged_train_summaries operations
        iteratively in proper order to to store training summary.
        N.B.: Must be implemented as per the requirement
        """
        pass

    def record_val_summary(self):
        """Runs the reset, accumulate and merged_val_summaries operations
        iteratively in proper order to to store training summary.
        N.B.: Must be implemented as per the requirement
        """
        pass

    def print(self,*args):
        print(*args)


class Logger(LoggerBase):

    def __init__(self,arg_dict):
        super().__init__( )
        self._init_all(arg_dict)

    def get_train_summary_op(self):
        """Returns the merged train summary operation for external use.
        e.g. running it along with training_step in the main training loop
        to avoid recomputations of batch summary variables for recording batch
        summary.
        """
        return self.objs["merged_train_summaries"]



    def record_train_summary(self, global_step,feed_dict={}):
        self.train_gen.reset_state_of_test_generator()
        self.session.run(self.ops["reset"])
        x, yt, bsz = self.train_gen.get_next_test_batch(self.bsz) # fecthes x, y, batch_size
        fdict = feed_dict
        while bsz!=None:
            feed_dict={self.model.inputs[0]:x,
                       self.model.true_outputs[0]:yt,
                       self.objs["current_sample_count"]:bsz}
            self.session.run(self.ops["accumulate"], feed_dict=feed_dict)
            x, yt, bsz = self.train_gen.get_next_test_batch(self.bsz)
        s = self.session.run(self.ops["merged_train_summaries"],feed_dict=fdict)
        self.objs["train_writer"].add_summary(s, global_step)
        self.objs["train_writer"].flush()

    def record_val_summary(self, global_step,feed_dict={}):
        self.val_gen.reset_state_of_test_generator()
        self.session.run(self.ops["reset"])
        x, yt, bsz = self.val_gen.get_next_test_batch(self.bsz)
        fdict = feed_dict
        while bsz!=None:
            feed_dict = {self.model.inputs[0]:x,
                         self.model.true_outputs[0]:yt,
                         self.objs["current_sample_count"]:bsz}
            self.session.run(self.ops["accumulate"], feed_dict=feed_dict)
            x, yt, bsz = self.val_gen.get_next_test_batch(self.bsz)
        s = self.session.run(self.ops["merged_val_summaries"],feed_dict=fdict)
        self.objs["val_writer"].add_summary(s,global_step)
        self.objs["val_writer"].flush()

if __name__ == '__main__':
    d = {"S":1,"K":2}
    for keys in d:
        print(keys)
    for keys in d.keys():
        print(keys)
    for keys in d.keys():
        print(keys, d[keys])