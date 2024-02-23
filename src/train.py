import tensorflow as tf
import numpy as np
import os

#TODO_ remove the network
from projection_net import ProjectionNet as net
from data import DataGenerator#TODO_ dev
X = np.linspace(100, 108, 8, dtype=np.int64)#TODO_ dev
Y = np.linspace(200, 108, 8, dtype=np.int64)#TODO_ dev
dgen = DataGenerator(X, Y, True)#TODO_ dev
def constant_lr(epoch):
    return 0.01

class TrainerBase:

    def __init__(self, arg_dict = {}):
        """Class to ease out the training process.
        *changing sessions may result in inconsistencies.
        *"""
        self. model = net()  #model which is to be trained #TODO_ remove net()--> None
        self.epochs = None  # epochs to run
        self.batch_size = None  # batch size
        self.load_pretrained = None  # flag: whether or not to load weights from file
        self.train_gen = dgen #TODO_ dev None  # A generator function to fetch training data
        self.val_gen = dgen #TODO_ dev None  # A generator function to fetch validation data
        self.initial_epoch = 1# Epoch number from where training should start
        self.epc = 0# epochs completed so far -- initial_epoch - 1
        self.batch_num = 0  # current batch number
        self.lr_schedule = constant_lr  # A function which takes epoch as input
                                        # and returns the value of the LR
        self.logger = None  # An instance of Logger class used for maintaing summaries
        self.log_dict = {}  # A dictonary of nodes to be recorded in summary
        self.metrics = {}  # A dictionary with key=Name of the metric, and value=
                           # metric function. These metrics will be calculated
                           # and recorded to generate the summary.
        self.cost_fns = {}  # A dictionary of cost function(s) to be used to train
                            # the network. If it is empty then model.costs will be used.
        self.optimizer = None  # optimizer to use e.g SGD
        self.model_summary_flag = False # flag: whether or not to include
                                         # summary ops defined by the model,
                                         # like weight histogram, etc. during
                                         # the calculation of summary metrics.
        self.batch_summ_step = None  # Number of batches processed after which-
                                  # the next batch-wise record is added to the
                                  # summary file.
        self.epoch_summ_step = None  # The number of epochs after which the next
                                  # epoch-wise training and validation summary
                                  # is written to the summary file.
        self.checkpoint_step = None# if None no checkpoints will be saved
        self.ops = {}  # A dictionary of operations. e.g. summary operations.
                       # Such as: 1. "batch_summary_op" 2. "val_summary_op"
                       # 3. "train_summary_op" etc.
        self.objs = {}  # A dictionary of the instances of various classes
                        # used by the TrainerBase for maintaining the flow of
                        # training and related opertaions.

    def start_training(self):
        """Initiates the training procedure"""
        self._set_up_training_environment()
        dsz = self.train_gen.get_dataset_size()
        max_batch_per_epoch = int(dsz/self.batch_size)
        assert(max_batch_per_epoch >= 1)
        if (dsz % self.batch_size) != 0:
            max_batch_per_epoch += 1
        self._start_training_loop(max_batch_per_epoch)


    def _start_training_loop(self, max_batch_per_epoch):
        self.batch_num = 0
        save_checkpoint = True
        while self.epc < self.epochs:
            self.batch_num += 1
            if self.batch_num % max_batch_per_epoch == 0:
                self.epc += 1
                save_checkpoint = True
            self._run_training_step()# batch summary is handled by this fn to avoid re-computations

            if self.checkpoint_step is not None:
                if self.epc % self.checkpoint_step == 0:
                    self._save_checkpoint()
                    save_checkpoint = False

            if self.epoch_summ_step is not None:
                if self.epc % self.epoch_summ_step == 0:
                    self._record_epoch_summary()

            if self.epc == self.epochs:  # training over
                if save_checkpoint and (self.checkpoint_step is not None):
                    self._save_checkpoint()
                    save_checkpoint = False
                if self.epoch_summ_step is not None:
                    if self.epc % self.epoch_summ_step != 0:
                        self._record_epoch_summary()

    def _save_checkpoint(self):
        pass

    def _run_training_step(self):
        """Executes training step and stores batch summary as required.
        """
        pass

    def stop_training(self):
        """Interface for stopping the training process and executing
        clean up process.
        """
        pass

    def _record_batch_summary(self):
        """Interface to call self.loggers functions to record
        batch-wise summary.
        """
        pass

    def _record_epoch_summary(self):
        """Interface to call self.logger's functions to record epoch-wise
        summary
        """
        pass

    def _set_up_cost_fn_and_optimizer(self):
        pass

    def _prepare_log_dict (self):
        """Prepare a list of nodes to be recorded in the summary.i.e. populates
        the list self.log_dict. It uses the functions in self.metrics to generate
        the elements of the self.log_list. It should be implemented as per the
        requirement.
        """
        pass

    def _set_up_training_environment(self):
        """Properly instantiates and initializes objects which are required for
        training the network.
        """
        self.epc = self.initial_epoch - 1

    def _set_up_summary_ops(self):
        """Adds summary operations to the graph, which will be used by the logger
        to record the summary data.
        """
        pass

    """
    Functions to be implemented in the derived class:
        self._record_batch_summary
        self._record_epoch_summary
        self._run_training_step #batch_summary should be handled in this function
        self._set_up_cost_fn_and_optimizer
        self._save_checkpoint
        self._set_up_training_environment
        self._prepare_log_dict
        self._set_up_summary_ops
    """


class Trainer(TrainerBase):
    """Trainer for single input and single output model.
    """
    def __init__(self,arg_dict={}):
        super().__init__(arg_dict)

    def _set_up_cost_fn_and_optimizer(self):
        pass

    def _prepare_log_dict(self):
        """Prepare a list of nodes to be recorded in the summary.i.e. populates
        the list self.log_dict. It uses the functions in self.metrics to gener-
        ate the elements of the self.log_dict.
        N.B.: It should be called after calling the self.model.build_network.
        """
        y_true = self.model.true_outputs[0]
        y_pred = self.model.outputs[0]
        for mx in self.metrics.keys():
            with tf.variable_scope(mx):
                self.log_dict[mx] = self.metrics[mx](y_true, y_pred)

        #self.model.build_network(tf.Session())# TODO_MUST REMOVE
        # IMPORTANT_ CALL AFTER ASSSIGNING self.model.costs
        if self.model.costs is not None:
            if isinstance(self.model.costs[0],dict):
                assert(len(self.model.costs)==1)
                for keys in self.model.costs[0]:
                    self.log_dict[keys] = self.model.costs[0][keys]
            else:
                self.log_dict["COST"] = self.model.costs[0]
        else:
            for keys in self.cost_fns.keys():
                with tf.variable_scope(keys):
                    self.log_dict[keys] = self.cost_fns[keys](y_true, y_pred)

    def _run_training_step(self):
        if self


if __name__ == "__main__":
    T = Trainer()
    def x (y_t, y_p):
        return tf.abs(y_t-y_p)
    def y (y_t, y_p):
        return tf.abs(y_t/y_p)
    T.metrics = {"X_metric":x, "Y_metric":y}
    print(T.log_dict)
    T._prepare_log_dict()
    print(T.log_dict)
    T.train_gen = dgen
    T.val_gen =dgen































