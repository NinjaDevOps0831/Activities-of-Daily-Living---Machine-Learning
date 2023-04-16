"""
    this is an abstract class of an implemented algorithm like hmm
    model is used for benchmarks
    model is also called from hmm events
"""

from hassbrain_algorithm.controller import Controller
from hassbrain_algorithm.datasets._dataset import DataRep
import joblib
import numpy as np
import copy

MD_FILE_NAME = "model_%s.joblib"

class Model(object):

    def __init__(self, name, controller):
        """

        Parameters
        ----------
        name
        controller

        Attributes
        ----------
        self.__dataset_repr (DataRep)
            is either RAW, LAST_FIRED or CHANGEPOINT
        """
        self._bench = None
        self._ctrl = controller # type: Controller
        self._model_name = MD_FILE_NAME
        self._dataset_repr = DataRep.RAW

        """
        these are callbacks that are can be accessed from third party classes
        for example the Benchmark to execute sth. during each training step.
        The model such as a hmm should call these methods after each step
        """
        self._callbacks = []

        """
        these hashmaps are used to decode and encode in O(1) the numeric based
        values of states and observations to the string based representations 
        """
        self._obs_lbl_hashmap = {}
        self._obs_lbl_rev_hashmap = {}
        self._state_lbl_hashmap = {}
        self._state_lbl_rev_hashmap = {}

        """
        Interpretability
        """
        self._skater_model = None
        self._explanator = None


    def get_data_repr(self):
        return self._dataset_repr

    def get_data_freq(self):
        """
        Returns
        -------
        str
            the frequence of the dataset the model was trained on
            e.g '1min', '30sec', ...
        """
        return self._dataset_conf['freq']

    def can_predict_next_obs(self):
        return NotImplementedError

    def can_predict_prob_devices(self):
        return NotImplementedError

    def encode_state_lbl(self, label):
        return self._state_lbl_hashmap[label]

    def decode_state_lbl(self, ide):
        return self._state_lbl_rev_hashmap[ide]

    def encode_obs_lbl(self, label):
        """
        returns the id of a sensor given a label
        :param label:
        :return:
        """
        return self._obs_lbl_hashmap[label]

    def decode_obs_lbl(self, ide):
        """
        retrieves the label given a sensor id
        :param id:
        :return:
        """
        return self._obs_lbl_rev_hashmap[ide]

    def get_state_lbl_lst(self):
         return list(self._state_lbl_hashmap.keys())

    def get_obs_lbl_lst(self):
        return list(self._obs_lbl_hashmap.keys())

    # todo flag for deletion
    def append_method_to_callbacks(self, callback_method):
        self._callbacks.append(callback_method)

    # todo flag for deletion
    def set_train_loss_callback(self):
        self._callbacks.append(self._train_loss_callback)

    def _train_loss_callback(self, *args):
        """
        hass to format the callback from the real model into an appropriate output for
        the benchmark method bench.train_loss_callback()
        :return:
        """
        raise NotImplementedError

    def register_benchmark(self, bench):
        self._bench = bench


    @classmethod
    def save_model(cls, model, path_to_folder, filename):
        """ saves this instance to a file
        Parameters
        ----------
        path_to_folder
        filename

        Returns
        -------

        """
        full_file_path = path_to_folder + "/" + filename
        import os
        if not os.path.exists(path_to_folder):
            os.makedirs(path_to_folder)
        #import dill
        #dill_file = open(full_file_path, "wb")
        #dill_file.write(dill.dumps(model))
        #dill_file.close()

        joblib.dump(model, full_file_path)

    @classmethod
    def load_model(cls, path_to_folder, filename):
        name = path_to_folder + "/" + filename
        return joblib.load(name)

        #import dill
        #dill_file = open(name, "rb")
        #model = dill.load(dill_file.read())
        #dill_file.close()
        #return model

    def set_name(self, name):
        self._name = name

    def get_name(self):
        return self._name

    def model_init(self, dataset, state_list=None):
        """
        initialize model on dataset
        :param dataset:
        :param state_list:
        :param location_data:
            a list conatining information how the smart home is set up
            loc_data = [ { "name" : "loc1", "activities" : ['cooking'],
            "devices" : ['binary_sensor.motion_floor', 'binary_sensor.motion_mirror'],
            }, ... }
        :return:
        """

        self.gen_hashmaps(dataset)

        if state_list is not None:
            self._state_list = dataset.encode
            # todo where is this used and what is state list set to
            raise ValueError
        else:
            self._state_list = dataset.get_state_list()

        self._observation_list = dataset.get_obs_list()
        self._dataset_repr = dataset.get_data_repr()

        self._dataset_conf = {
            'freq': dataset.get_freq(),
            'repr': dataset.get_data_repr(),
            'obs_type': 'bernoulli'
        }

        self._model_init(dataset)

    def gen_hashmaps(self, dataset):
        self._obs_lbl_hashmap = dataset.get_obs_lbl_hashmap()
        self._obs_lbl_rev_hashmap = dataset.get_obs_lbl_reverse_hashmap()
        self._state_lbl_hashmap = dataset.get_state_lbl_hashmap()
        self._state_lbl_rev_hashmap = dataset.get_state_lbl_reverse_hashmap()
        self.K = len(self._state_lbl_hashmap)
        self.D = len(self._obs_lbl_rev_hashmap)

    def are_hashmaps_created(self):
        return self._obs_lbl_hashmap is None \
               or self._obs_lbl_rev_hashmap is None \
               or self._state_lbl_hashmap is None \
               or self._state_lbl_rev_hashmap is None


    def register_act_info(self, act_data):
        self._act_data = self._encode_act_data(act_data)

    def register_loc_info(self, loc_data):
       self._loc_data = self._encode_location_data(loc_data)

    def _encode_act_data(self, act_data):
        """
        encodes the device names and activity names into numbers that the models
        can understand
        :param loc_data:
            is a list of loc data
            example:
            [ { "name" : "loc1", "activities" : ['cooking'],
            "devices" : ['binary_sensor.motion_floor', 'binary_sensor.motion_mirror'],
            },  ... ]
        :return:
            a list of location data, with the same structure correct encoded labels
             [ { "name" : "loc1",
                "activities" : [ 1 ],
                "devices" : [3, 7],
            },  ... ]
        """
        for activity in act_data:
            activity["name"] = self.encode_state_lbl(activity["name"])
        return act_data


    def _encode_location_data(self, loc_data):
        """
        encodes the device names and activity names into numbers that the models
        can understand
        :param loc_data:
            is a list of loc data
            example:
            [ { "name" : "loc1", "activities" : ['cooking'],
            "devices" : ['binary_sensor.motion_floor', 'binary_sensor.motion_mirror'],
            },  ... ]
        :return:
            a list of location data, with the same structure correct encoded labels
             [ { "name" : "loc1",
                "activities" : [ 1 ],
                "devices" : [3, 7],
            },  ... ]
        """
        for location in loc_data:
            new_act_list = []
            for activity in location['activities']:
                # todo the use of str vs. int is a hint that there is an
                # inconsistent use in the encoding and decoding in state labels
                new_act_list.append(str(self.encode_state_lbl(activity)))
            location['activities'] = new_act_list

            new_dev_list = []
            for device in location['devices']:
                new_dev_list.append(self.encode_obs_lbl(device))
            location['devices'] = new_dev_list
        return loc_data

    def _model_init(self, dataset, location_data):
        """
        this method has to be overriden by child classes
        :return:
        """
        raise NotImplementedError

    def predict_state_sequence(self, test_y):
        """
        for a given sequence the model should predict the underlying activities
        Parameters
        ----------
        test_y  array_like (... K)

        Returns
        -------
        pred_x  array_like (... K)
            the predicted activities
        """
        raise NotImplementedError

    def predict_obs_sequence(self, test_y):
        """
        for a given sequence the model should predict the next probable observation y_{t+1}
        at y_t
        Parameters
        ----------
        test_y  array_like (... K)

        Returns
        -------
        pred_y  array_like (... K)
        """
        raise NotImplementedError

    def train(self, dataset):
        """
         gets a dataset and trains the model on the data
         Important !!!!
         during the training the hashmaps of the model have to be filled up
        :param dataset:
        :return:
            the loss per iteration list
        """

        y_train = dataset.get_train_data()
        test_z, test_x = dataset.get_all_labeled_data() # states, obs
        lls = self._train(dataset)
        return lls


    def _train(self, dataset):
        raise NotImplementedError

    def get_state(self, seq):
        """
        returns the state the model is in given an observation sequence
        :return:
        """
        raise NotImplementedError

    def save_visualization(self, path_to_file):
        """
        save a visualization of the model to the given filepath
        :param path_to_file:
        :return:
        """
        raise NotImplementedError

    def draw(self):
        """
         somehow visualize the model
        :return: an image png or jpg
        """
        raise NotImplementedError

    def classify_multi(self, obs_seq, as_dict=True):
        """
        gets an observation sequence (at most times this is a window) and returns
        the most likely states
        :param obs_seq:
        :return:
        np array or dict
            with the most likely state
        """
        # encode obs_seq
        #obs_seq = self.obs_lbl_seq2enc_obs_seq(obs_seq)
        # array full of tuples with state_label and corresp. score
        import numpy as np
        scores = self._classify_multi(obs_seq)
        assert isinstance(scores, np.ndarray) and len(scores.shape) == 1

        if not as_dict:
            return scores
        else:
            act_score_dict = {}
            # decode state seq
            for i, score in enumerate(scores):
                label = self._state_lbl_rev_hashmap[i]
                act_score_dict[label] = score
            return act_score_dict

    def classify(self, obs_seq):
        """
        gets an observation sequence (at most times this is a window) and returns
        the most likely states
        Parameters
        ----------
        obs_seq np.ndarray
            list of np arrays containing the observations vectors
            e.g [ [0,1,0,0,1,...,0] , [0,0,1,0,...,1], ... ]

        Returns
        -------
        best_state (string)
            the most probable state the sequence is in

        """
        # encode obs_seq
        #obs_seq = self.obs_lbl_seq2enc_obs_seq(obs_seq)
        pred_state = self._classify(obs_seq)

        # decode state seq
        res = self._state_lbl_rev_hashmap[pred_state]
        return res

    def sample(self, n, obs_seq=None):
        """ samples
        Parameters
        ----------
        obs_seq
            optional observations which are used to sample stuff

        n   integer
            the amount of timesteps to generate

        Returns
        -------
        dec_states nd.array of strings

        dec_obs nd.array of strings
        """
        enc_states, enc_obs = self._sample(n, obs_seq)
        dec_states = self.decode_state_lbl_list(enc_states)
        dec_obs = self.decode_obs_lbl_list(enc_obs)
        return dec_states, dec_obs

    def decode_obs_lbl_list(self, lst):
        #return self._map_nparr_to_lbls(self._obs_lbl_rev_hashmap, lst)
        pass

    def decode_state_lbl_list(self, lst):
        return self._map_nparr_to_lbls(self._state_lbl_rev_hashmap, lst)


    def _map_nparr_to_lbls(self, dict, arr):
        """
        gets a encoded list and returns a list of labels
        Parameters
        ----------
        lst     list or np.ndarray
        Returns
        -------
            np.ndarray
        """
        N = len(arr)
        import numpy as np
        res = np.zeros((N), dtype=object)
        for i in range(N):
            val = arr[i]
            dec_val = dict[val]
            res[i] = dec_val
        return res


    def _sample(self, n, obs_seq=None):
        raise NotImplementedError

    def predict_next_obs(self, obs_seq):
        """
        predict the most probable device to change its state
        :param args:
        :return:
            label (string)
            the label of the device to change its state
        """
        #obs_seq = self.obs_lbl_seq2enc_obs_seq(obs_seq)
        last_obs = obs_seq[-1:][0]
        idx_next_obs = self._predict_next_obs(obs_seq)
        label = self._obs_lbl_rev_hashmap[idx_next_obs]
        return label

    def predict_prob_xnp1(self, obs_seq):
        """
        computes the probabilities of all observations to be the next
        :param obs_seq:
        :return:
            dictionary like: "{ 'sensor_name' : { 0 : 0.123, 1: 0.789 } , ... }"
        """
        #obs_seq = self.obs_lbl_seq2enc_obs_seq(obs_seq)
        arr = self._predict_prob_xnp1(obs_seq)
        res_dict = copy.deepcopy(self._obs_lbl_hashmap)
        for i, prob in enumerate(arr):
            label = self._obs_lbl_rev_hashmap[i]
            res_dict[label] = prob
        return res_dict

    def _classify(self, obs_seq):
        raise NotImplementedError

    def _classify_multi(self, obs_seq):
        raise NotImplementedError

    def _predict_next_obs(self, obs_seq):
        """
        has to return an array containing all the probabilities of the observations
        to be next
        :param obs_seq:
        :return:
        """
        raise NotImplementedError

    def _predict_prob_xnp1(self, obs_seq):
        raise NotImplementedError
