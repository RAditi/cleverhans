"""
Random Pertubations in L_inf
"""

import warnings

import numpy as np
import tensorflow as tf

from cleverhans.attacks.attack import Attack
from cleverhans.compat import reduce_max, reduce_sum, softmax_cross_entropy_with_logits
from cleverhans import utils_tf


class FastGradientMethod(Attack):
  """
  Just adding random perturbations within the attack model 

  :param model: cleverhans.model.Model
  :param sess: optional tf.Session
  :param dtypestr: dtype of the data
  :param kwargs: passed through to super constructor
  """

  def __init__(self, model, sess=None, dtypestr='float32', **kwargs):
    """
    Create a RandomPerturb instance.
    Note: the model parameter should be an instance of the
    cleverhans.model.Model abstraction provided by CleverHans.
    """

    super(RandomPerturb, self).__init__(model, sess, dtypestr, **kwargs)
    self.feedable_kwargs = ('eps', 'y', 'y_target', 'clip_min', 'clip_max')
    self.structural_kwargs = ['ord', 'sanity_checks']

  def generate(self, x, **kwargs):
    """
    Returns the graph for random perturbations.

    :param x: The model's symbolic inputs.
    :param kwargs: See `parse_params`
    """
    # Parse and save attack-specific parameters
    assert self.parse_params(**kwargs)

    labels, _nb_classes = self.get_or_guess_labels(x, kwargs)

    # Creating random signed vector as perturbation 
    eta = tf.random_uniform(tf.shape(x),
                              tf.cast(-1, x.dtype),
                              tf.cast(1, x.dtype),
                              dtype=x.dtype)

    perturbation = tf.sign(eta)*self.eps
    return x + perturbation


  def parse_params(self,
                   eps=0.3,
                   ord=np.inf,
                   y=None,
                   y_target=None,
                   clip_min=None,
                   clip_max=None,
                   sanity_checks=True,
                   **kwargs):
    """
    Take in a dictionary of parameters and applies attack-specific checks
    before saving them as attributes.

    Attack-specific parameters:

    :param eps: (optional float) attack step size (input variation)
    :param ord: (optional) Order of the norm (mimics NumPy).
                Possible values: np.inf, 1 or 2.
    :param y: (optional) A tensor with the true labels. Only provide
              this parameter if you'd like to use true labels when crafting
              adversarial samples. Otherwise, model predictions are used as
              labels to avoid the "label leaking" effect (explained in this
              paper: https://arxiv.org/abs/1611.01236). Default is None.
              Labels should be one-hot-encoded.
    :param y_target: (optional) A tensor with the labels to target. Leave
                     y_target=None if y is also set. Labels should be
                     one-hot-encoded.
    :param clip_min: (optional float) Minimum input component value
    :param clip_max: (optional float) Maximum input component value
    :param sanity_checks: bool, if True, include asserts
      (Turn them off to use less runtime / memory or for unit tests that
      intentionally pass strange input)
    """
    # Save attack-specific parameters

    self.eps = eps
    self.ord = ord
    self.y = y
    self.y_target = y_target
    self.clip_min = clip_min
    self.clip_max = clip_max
    self.sanity_checks = sanity_checks

    if self.y is not None and self.y_target is not None:
      raise ValueError("Must not set both y and y_target")
    # Check if order of the norm is acceptable given current implementation
    if self.ord not in [np.inf, int(1), int(2)]:
      raise ValueError("Norm order must be either np.inf, 1, or 2.")

    if len(kwargs.keys()) > 0:
      warnings.warn("kwargs is unused and will be removed on or after "
                    "2019-04-26.")

    return True
   