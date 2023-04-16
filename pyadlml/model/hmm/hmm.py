import math

from hassbrain_algorithm.models._model import Model
import autograd.numpy as np
import autograd.numpy.random as npr
from scipy.stats import nbinom
import matplotlib.pyplot as plt
import ssm
from ssm.util import rle, find_permutation

class HMM(Model):

    def __init__(self, controller):
        #self._cm = controller # type: Controller
        # training parameters
        self._training_steps = 500
        self._epsilon = None
        self._use_q_fct = False

        Model.__init__(self, "test", controller)

    def __str__(self):
        if self._hmm is None:
            return "hmm has to be inits"
        else:
            s = "--"*10 + "\n"
            s += "States: " + str(self._hmm.K) + "\n"
            s += "Obs:\t " + str(self._hmm.D) + "\n"
            s += "pi:\t " + str(self._hmm.init_state_distn) + "\n"
            s += "Trans:\t " + str(self._hmm.transitions) + "\n"
            s += "M:\t " + str(self._hmm.M) + "\n"
            return s

    def _model_init(self, dataset):
        K = len(self._state_list)       # number of discrete states
        D = len(self._observation_list) # dimension of the observation
        self._hmm = ssm.HMM(K, D, observations='bernoulli')


    def save_visualization(self, path_to_file):
        """
        save a visualization of the model to the given filepath
        :param path_to_file:
        :return:
        """
        pass

    def save_visualization_helper_decode_labels(self, label):
        """
        :return:
        """
        return self.decode_state_lbl(label)


    def get_train_loss_plot_y_label(self):
        if self._use_q_fct:
            return 'Q(Theta, Theta_old)'
        else:
            return 'P(X|Theta)'


    def predict_state_sequence(self, test_y):
        """

        :param test_y:
        :return:
        """
        pred_x = self._hmm.most_likely_states(test_y)
        return pred_x

    def predict_obs_sequence(self, test_y):
        """

        :param test_y:
        :return:
        """
        pred_y = self._hmm.sample(test_y)
        return pred_y


    def can_predict_next_obs(self):
        return False

    def can_predict_prob_devices(self):
        return False

    def draw(self, act_retrieval_meth):
        self._hmm.plot()
        return self._hmm.generate_graphviz_dot_ext_lbl(act_retrieval_meth)
        #vg.render('test.gv', view=True)

    def _train_loss_callback(self, hmm, loss, *args):
        # todo in log models convert to normal Probability
        """
        :param hmm:
        :param loss:
        :param args:
        :return:
        """
        # this is due to the the loss param is actually the likelihood of the P(X|Model)
        # therefore the loss can be 1 - P(X|Model)
        loss = 1-loss
        self._bench.train_loss_callback(hmm, loss)


    def train(self, dataset, args):
        y_train = dataset.get_train_data()
        hsmm_em_lls = self._hmm.fit(
            y_train,
            method="em",
            num_em_iters=self._training_steps)

        test_x, test_y = dataset.get_all_labeled_data()
        # todo uncomment line below, very important !!!!!!!!
        self.assign_states(test_x, test_y)
        return hsmm_em_lls

    def assign_states(self, true_z, true_y):
        """
        assigns the unordered hidden states of the trained model (on true_y)
        to the most probable state labels in alignment of true_z
        :param true_z
            the true state sequence of a labeled dataset
        :param true_y
            the true corresp. observation sequence of a labeled dataset

        assign
           z = true state seq [1,2,1,....,]
           tmp3 = pred. state seq [3,4,1,2,...,]
           match each row to different column in such a way that corresp
           sum is minimized
           select n el of C, so that there is exactly one el.  in each row
           and one in each col. with min corresp. costs


        match states [1,2,...,] of of the
        :return:
            None
        """
        # Plot the true and inferred states
        tmp1 = self._hmm.most_likely_states(true_y)
        # todo temporary cast to int64 remove for space saving solution
        # todo as the amount of states only in range [0, 30
        true_z = true_z.astype(np.int64)
        tmp2 = find_permutation(true_z, tmp1)
        self._hmm.permute(tmp2)





#-------------------------------------------------------------------
    # RT Node stuff

    def _classify(self, obs_seq):
        """
        get the most probable state/activity
        :param obs_seq:
        :return:
        """
        assert len(obs_seq[0]) == self._hmm.D
        state_seq = self._hmm.most_likely_states(obs_seq)
        pred_state = state_seq[-1:][0]
        return pred_state

    def _classify_multi(self, obs_seq):
        """
        computes the last omega slice of viterbi which is
        equivalent to
        :param obs_seq:
        :return:
        """
        tmp = self._hmm.filter(obs_seq)
        last_alpha = tmp[-1:][0]


        assert math.isclose(last_alpha.sum(), 1.)

        K = self._hmm.K

        score_dict = {}
        for i in range(K):
            label = i
            score = last_alpha[i]
            score_dict[label] = score
        return score_dict



    def _predict_next_obs(self, obs_seq):
        pre_states = self._hmm.most_likely_states(obs_seq)
        next_obs = self._hmm.sample(1, prefix=pre_states)
        next_obs = next_obs[0]
        return next_obs

    def _predict_prob_xnp1(self, obs_seq):
        # todo here hsmm has to be modified, tap in the process and
        # return the probabilities
        res = self._hmm.sample(obs_seq)
        return res

#class BernoulliHMM_Cat(BernoulliHMM):
#    def __init__(self, controller):
#       BernoulliHMM.__init__(self, controller)
#
#    def _model_init(self, dataset):
#        K = len(self._state_list)       # number of discrete states
#        D = len(self._observation_list) # dimension of the observation
#        self._hmm = ssm.HMM(K, D, observations='categorical')

