# -*- coding: utf-8 -*-
"""
Created on Tue, 20/Feb/2018

@author: Yoshimasa Kubo
"""

from pylab import *
from scipy import io
import matplotlib.pyplot as plt
import os

## for deep 
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_input
from enum import Enum
from tensorflow.python.framework import ops
import sys

WEIGHT_DECAY = False
#STATE_NUM    = 128
NUM_OUT      = 10
learning_rate_conv = 0.001

slope_annealing_rate = 1.04
slope = 1.0
early_stopping_step = 20000
loss_acc_flg = 1 # 0: loss check, 1: accuracy check

input_size      = 784
image_size      = 28
image_depth     = 1
class_size      = 10
batch_size      = 50

filter_size       = [5,5]
previous_channels = [1, 32]
channels          = [32, 64]
LAYER_1           = 512 
LAYER_2           = 1024
LAYER_LINEAR      = 100

argvs = sys.argv
dir_num        =  [float(argvs[1])][0]
STATE_NUM      =  [float(argvs[2])][0]
directory      = 'sw_2cnn_bn_tanh3_wow_er_'+ str(STATE_NUM) + '_' + str(float(dir_num))

mnist = mnist_input.read_data_sets('MNIST_data', one_hot=True)

# weight_decay
regularizer_ = None
if WEIGHT_DECAY:
  regularizer_ = tf.contrib.layers.l2_regularizer(L2)

alpha = 0.3
def lrelu(x):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.
    # Returns
        A tensor.
    """
    return tf.convert_to_tensor(x, dtype=dtype)

def hard_sigmoid(x):
    """Segment-wise linear approximation of sigmoid.
    Faster than sigmoid.
    Returns `0.` if `x < -2.5`, `1.` if `x > 2.5`.
    In `-2.5 <= x <= 2.5`, returns `0.2 * x + 0.5`.
    # Arguments
        x: A tensor or variable.
    # Returns
        A tensor.
    """
    x = (0.2 * x) + 0.5
    zero = _to_tensor(0., x.dtype.base_dtype)
    one = _to_tensor(1., x.dtype.base_dtype)
    x = tf.clip_by_value(x, zero, one)
    return x

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

def layer_linear(inputs, shape, scope='linear_layer'):
    with tf.variable_scope(scope):
      if WEIGHT_DECAY:
        w = tf.get_variable('w',shape, initializer=tf.random_uniform_initializer(minval=-1, maxval=1),  regularizer=regularizer_)
      else:
        w = tf.get_variable('w',shape, initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
        
      b = tf.get_variable('b',shape[-1:], initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
    return tf.matmul(inputs,w) + b

def layer_softmax(inputs, shape, scope='softmax_layer'):
    with tf.variable_scope(scope):
        w = tf.get_variable('w',shape, initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
        b = tf.get_variable('b',shape[-1:], initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
    return tf.nn.softmax(tf.matmul(inputs,w) + b)

def accuracy(y, pred):
    correct = tf.equal(tf.argmax(y,1), tf.argmax(pred,1))
    return tf.reduce_mean(tf.cast(correct, tf.float32))

def plot_n(data_and_labels, lower_y = 0., title="Learning Curves"):
    fig, ax = plt.subplots()
    for data, label in data_and_labels:
        ax.plot(range(0,len(data)*100,100),data, label=label)
    ax.set_xlabel('Training steps')
    ax.set_ylabel('Accuracy')
    ax.set_ylim([lower_y,1])
    ax.set_title(title)
    ax.legend(loc=4)
    plt.show()

class StochasticGradientEstimator(Enum):
    ST = 0
    REINFORCE = 1 
    
def binaryRound(x):
    """
    Rounds a tensor whose values are in [0,1] to a tensor with values in {0, 1},
    using the straight through estimator for the gradient.
    """
    g = tf.get_default_graph()

    with ops.name_scope("BinaryRound") as name:
      with g.gradient_override_map({"Round": "Identity"}):
          return tf.round(x, name=name)

      # For Tensorflow v0.11 and below use:
      #with g.gradient_override_map({"Floor": "Identity"}):
      #    return tf.round(x, name=name)
        
def bernoulliSample(x):
  """
  Uses a tensor whose values are in [0,1] to sample a tensor with values in {0, 1},
  using the straight through estimator for the gradient.

  E.g.,:
  if x is 0.6, bernoulliSample(x) will be 1 with probability 0.6, and 0 otherwise,
  and the gradient will be pass-through (identity).
  """
  g = tf.get_default_graph()

  with ops.name_scope("BernoulliSample") as name:
    with g.gradient_override_map({"Ceil": "Identity","Sub": "BernoulliSample_ST"}):
      #return tf.sign(x - tf.random_uniform(tf.shape(x)), name=name)
      return tf.ceil(x - tf.random_uniform(tf.shape(x)), name=name)

@ops.RegisterGradient("BernoulliSample_ST")
def bernoulliSample_ST(op, grad):
  return [grad, tf.zeros(tf.shape(op.inputs[1]))]
        
def passThroughSigmoid(x, slope=1):
  """Sigmoid that uses identity function as its gradient"""
  g = tf.get_default_graph()
  with ops.name_scope("PassThroughSigmoid") as name:
    with g.gradient_override_map({"Sigmoid": "Identity"}):
      return tf.sigmoid(x, name=name)
      #return hard_sigmoid(x)

def binaryStochastic_ST(x, slope_tensor=None, pass_through=True, stochastic=True):
  """
  Sigmoid followed by either a random sample from a bernoulli distribution according
  to the result (binary stochastic neuron) (default), or a sigmoid followed by a binary
  step function (if stochastic == False). Uses the straight through estimator.
  See https://arxiv.org/abs/1308.3432.

  Arguments:
  * x: the pre-activation / logit tensor
  * slope_tensor: if passThrough==False, slope adjusts the slope of the sigmoid function
      for purposes of the Slope Annealing Trick (see http://arxiv.org/abs/1609.01704)
  * pass_through: if True (default), gradient of the entire function is 1 or 0;
      if False, gradient of 1 is scaled by the gradient of the sigmoid (required if
      Slope Annealing Trick is used)
  * stochastic: binary stochastic neuron if True (default), or step function if False
  """
  if slope_tensor is None:
      slope_tensor = tf.constant(1.0)

  if pass_through:
      p = passThroughSigmoid(x)
  else:
      p = tf.sigmoid(slope_tensor*x)

  if stochastic:
      return bernoulliSample(p)
  else:
      return binaryRound(p)      

def binaryStochastic_REINFORCE(x, stochastic = True, loss_op_name="loss_by_example"):
  """
  Sigmoid followed by a random sample from a bernoulli distribution according
  to the result (binary stochastic neuron). Uses the REINFORCE estimator.
  See https://arxiv.org/abs/1308.3432.

  NOTE: Requires a loss operation with name matching the argument for loss_op_name
  in the graph. This loss operation should be broken out by example (i.e., not a
  single number for the entire batch).
  """
  g = tf.get_default_graph()

  with ops.name_scope("BinaryStochasticREINFORCE"):
    with g.gradient_override_map({"Sigmoid": "BinaryStochastic_REINFORCE", "Ceil": "Identity"}):
      p = tf.sigmoid(x)

      reinforce_collection = g.get_collection("REINFORCE")
      if not reinforce_collection:
        g.add_to_collection("REINFORCE", {})
        reinforce_collection = g.get_collection("REINFORCE")
      reinforce_collection[0][p.op.name] = loss_op_name

      return tf.ceil(p - tf.random_uniform(tf.shape(x)))
          
@ops.RegisterGradient("BinaryStochastic_REINFORCE")
def _binaryStochastic_REINFORCE(op, _):
  """Unbiased estimator for binary stochastic function based on REINFORCE."""
  loss_op_name = op.graph.get_collection("REINFORCE")[0][op.name]
  loss_tensor = op.graph.get_operation_by_name(loss_op_name).outputs[0]


  sub_tensor = op.outputs[0].consumers()[0].outputs[0] #subtraction tensor
  ceil_tensor = sub_tensor.consumers()[0].outputs[0] #ceiling tensor

  outcome_diff = (ceil_tensor - op.outputs[0])

  # Provides an early out if we want to avoid variance adjustment for
  # whatever reason (e.g., to show that variance adjustment helps)
  if op.graph.get_collection("REINFORCE")[0].get("no_variance_adj"):
      return outcome_diff * tf.expand_dims(loss_tensor, 1)

  outcome_diff_sq = tf.square(outcome_diff)
  outcome_diff_sq_r = tf.reduce_mean(outcome_diff_sq, reduction_indices=0)
  
  outcome_diff_sq_loss_r = tf.reduce_mean(outcome_diff_sq * tf.expand_dims(loss_tensor, 1),
                                          reduction_indices=0)


  L_bar_num = tf.Variable(tf.zeros(outcome_diff_sq_r.get_shape()), trainable=False)

  L_bar_den = tf.Variable(tf.ones(outcome_diff_sq_r.get_shape()), trainable=False)


  #Note: we already get a decent estimate of the average from the minibatch
  decay = 0.95
  train_L_bar_num = tf.assign(L_bar_num, L_bar_num*decay +\
                                          outcome_diff_sq_loss_r*(1-decay))
  train_L_bar_den = tf.assign(L_bar_den, L_bar_den*decay +\
                                          outcome_diff_sq_r*(1-decay))


  with tf.control_dependencies([train_L_bar_num, train_L_bar_den]):
    L_bar = train_L_bar_num/(train_L_bar_den+1e-4)
    L = tf.tile(tf.expand_dims(loss_tensor,1), tf.constant([1,L_bar.get_shape().as_list()[0]]))
    return outcome_diff * (L - L_bar)
      
def binary_wrapper(pre_activations_tensor, estimator=StochasticGradientEstimator.ST, stochastic_tensor=tf.constant(True), pass_through=True, slope_tensor=tf.constant(1.0)):
  """
  Turns a layer of pre-activations (logits) into a layer of binary stochastic neurons

  Keyword arguments:
  *estimator: either ST or REINFORCE
  *stochastic_tensor: a boolean tensor indicating whether to sample from a bernoulli
      distribution (True, default) or use a step_function (e.g., for inference)
  *pass_through: for ST only - boolean as to whether to substitute identity derivative on the
      backprop (True, default), or whether to use the derivative of the sigmoid
  *slope_tensor: for ST only - tensor specifying the slope for purposes of slope annealing
      trick
  """

  if estimator == StochasticGradientEstimator.ST:
    if pass_through:
      return tf.cond(stochastic_tensor,
        lambda: binaryStochastic_ST(pre_activations_tensor),
        lambda: binaryStochastic_ST(pre_activations_tensor, stochastic=False))
    else:
      return tf.cond(stochastic_tensor,
        lambda: binaryStochastic_ST(pre_activations_tensor, slope_tensor = slope_tensor, pass_through=False),
        lambda: binaryStochastic_ST(pre_activations_tensor, slope_tensor = slope_tensor, pass_through=False, stochastic=False))
      
  elif estimator == StochasticGradientEstimator.REINFORCE:
      # binaryStochastic_REINFORCE was designed to only be stochastic, so using the ST version
      # for the step fn for purposes of using step fn at evaluation / not for training
    return tf.cond(stochastic_tensor, lambda: binaryStochastic_REINFORCE(pre_activations_tensor), lambda: binaryStochastic_ST(pre_activations_tensor, stochastic=False))

  else:
    raise ValueError("Unrecognized estimator.")

def batch_norm(training, out, shape, scope=1, layer_type=0, decay=0.9):
  ema = tf.train.ExponentialMovingAverage(decay=decay)
  a   = tf.Variable(tf.ones(shape, dtype=tf.float32))
  b   = tf.Variable(tf.zeros(shape, dtype=tf.float32))
  mu  = None
  var = None
  if layer_type == 0:
    mu,var = tf.nn.moments(out,[0,1,2])
  else:
    mu,var = tf.nn.moments(out,[0])
    
  def mean_var_with_update():
    ema_apply_op = ema.apply([mu, var])
    with tf.control_dependencies([ema_apply_op]):
        return tf.identity(mu), tf.identity(var)
    
  mu, var = tf.cond(training, mean_var_with_update, lambda: (ema.average(mu), ema.average(var)))
    
  return tf.nn.batch_normalization(out, mu, var, b, a, 1E-4, name='batch_norm' + str(scope))

# creating linear layer
def linear_layer(input_l, nin, nout, name, layer, activation):

  if WEIGHT_DECAY:
    weight = tf.get_variable(name=name, shape=(nin, nout), initializer=tf.random_uniform_initializer(minval=-1, maxval=1), regularizer=regularizer_)
  else:
    weight = tf.get_variable(name=name, shape=(nin, nout), initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
    
  linear_bias   = tf.Variable(tf.constant(0.1, shape=[nout]), name='bias' + str(layer))

  out_pre  = tf.matmul(input_l, weight)
  

  out  = out_pre + linear_bias
  
  # Need activation (relu) ?
  if activation:
    out = tf.nn.relu(out)
    #out = lrelu(out)

    
  return out

# creating convolutional + batch norm + maxpooling layers
def conv_layer(x_input, filter_size, strides, padding, name_f, name_conv, layer, activation, previous_channels, channels, pool, training):

  if layer == 1:
    x_input = tf.reshape(x_input, [-1, image_size, image_size, 1])
  # creating weight
  #if WEIGHT_DECAY:
  if layer == 2:
    filter_l = tf.get_variable(name=name_f, shape=[filter_size, filter_size, previous_channels, channels], initializer=tf.random_uniform_initializer(minval=-1, maxval=1), regularizer=regularizer_)
  else:
    filter_l = tf.get_variable(name=name_f, shape=[filter_size, filter_size, previous_channels, channels], initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
    
  bias = tf.Variable(tf.constant(0.1, shape=[channels], name='bias' + str(layer)))
  
  # conv layer
  conv = tf.nn.conv2d(x_input, filter_l,  strides=strides, padding=padding, name=name_conv)

  out  = batch_norm(training, conv, channels, layer)
 
  # Need Pooling ?
  if pool:  k,s=pool; out = tf.nn.max_pool(out, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding=padding)
  
  # Need activation (elu, relu, or tanh) ?
  if activation:
    out = tf.nn.relu(out)
  
  return out


def ind_input_multiply(inputs):
  input_    = tf.reshape(inputs[0], (1, input_size))
  weight_   = inputs[1]
  
  input_ = tf.reshape(input_, [-1, image_size, image_size, 1])
  return tf.nn.conv2d(input_, weight_,  strides=[1, 2, 2, 1], padding='VALID')#, name=name_conv)


def ind_2nd_multiply(inputs):
  input_    = inputs[0]
  weight_   = inputs[1]
  
  input_ = tf.expand_dims(input_, axis=0)
  return tf.nn.conv2d(input_, weight_,  strides=[1, 2, 2, 1], padding='VALID')#, name=name_conv)



def linear_input_multiply(inputs):
  input_    = tf.expand_dims(inputs[0], axis=0)
  weight_   = inputs[1]

  return   tf.matmul(input_, weight_) 

def linear_inference(input_l, output_shape, scope_ , activation=True):
   
  out_l = layer_linear(input_l, (input_l.get_shape()[1], output_shape), scope='layer_' + str(scope_))
  
  if activation:
    out_l = tf.tanh(out_l)
    
  return out_l

def conv2d_inference(input_l, layer_type, filter_size_l, channels_l, name_f, name_conv, conv_num, image_depth_l, training, batch_norm_num):
  conv1 = conv_layer(input_l, filter_size_l, [1, 2, 2, 1], 'VALID', name_f, name_conv, conv_num, True, image_depth_l, channels_l, False, training)

  conv2  = conv1
  shape2_ = conv2.get_shape()

  h_conv2_flat = tf.reshape(conv2, [-1, int(shape2_[1])*int(shape2_[2])*channels_l])
  out_conv1    = linear_layer(h_conv2_flat, int(shape2_[1])*int(shape2_[2])*channels_l, layer_type, 'linear1', 3, False)
  out_conv1    = batch_norm(training, out_conv1, layer_type, batch_norm_num, 1)
  out_conv1    = tf.nn.relu(out_conv1)

  return out_conv1

################ main inference ########################
def joint_inference(x_input, training, slope, dropout):
  
   
  with tf.variable_scope('1st_layer'):
    # first layer 
    filter_l = tf.get_variable(name='cnn_layer1', shape=[filter_size[0], filter_size[0], previous_channels[0], channels[0]], initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
    bias = tf.Variable(tf.constant(0.1, shape=[channels[0]], name='bias1'))

    # first input (weights)
    filter_l = tf.expand_dims(filter_l, axis=0)
    all_weight_1 = tf.tile(filter_l, [tf.shape(x_input)[0],1, 1, 1, 1])
    all_weight_1 = tf.reshape(all_weight_1, (tf.shape(x_input)[0], channels[0], filter_size[0]*filter_size[0]))

    
    # second input (images) 
    all_input_1      = conv2d_inference(x_input, LAYER_1, filter_size[0], channels[0], 'conv1', 'conv1_layer', 1, image_depth, training, 11)
   
    all_inputs_1 = all_input_1

    # stochastic layer
    with tf.variable_scope('multiplication') as scope:
      output_state_1    = layer_linear(all_inputs_1, (all_inputs_1.get_shape()[1], STATE_NUM), scope='layer_pre_pre')
      output_state_1    = tf.tanh(output_state_1) #elu or relu?
      
      output_state_1 = tf.nn.dropout(output_state_1, dropout)
      
      pre_activations_1 = layer_linear(output_state_1, (output_state_1.get_shape()[1], filter_size[0]*filter_size[0]*previous_channels[0]*channels[0]), scope='layer_pre')
      pre_st_weight_1   = binary_wrapper(pre_activations_1, estimator =  StochasticGradientEstimator.ST, pass_through = True, stochastic_tensor = tf.constant(True), slope_tensor = slope)
      
      pre_st_weight_1   = tf.reshape(pre_st_weight_1, (tf.shape(pre_st_weight_1)[0], filter_size[0], filter_size[0], previous_channels[0], channels[0]))
      inference_fn_w    = lambda input_w: tf.multiply(filter_l, input_w)
      st_weight_1       = tf.map_fn(inference_fn_w, pre_st_weight_1, dtype=tf.float32, swap_memory=True)
      st_weight_1       = tf.squeeze(st_weight_1, axis=1)
  
      input_ = (x_input, st_weight_1)
      inference_fn_out = lambda inp_: ind_input_multiply(inp_)
      out_1 = tf.map_fn(inference_fn_out, input_, dtype=tf.float32, swap_memory=True)
      out_1  = tf.squeeze(out_1, axis=1)
      out_1  = batch_norm(training, out_1, channels[0], 1)
      out_1  = tf.nn.relu(out_1) #elu or relu?
      
  with tf.variable_scope('2nd_layer'):
    # second layer 
    filter_l2 = tf.get_variable(name='cnn_layer2', shape=[filter_size[1], filter_size[1], previous_channels[1], channels[1]], initializer=tf.random_uniform_initializer(minval=-1, maxval=1)) #, initializer=tf.contrib.layers.xavier_initializer())
    #filter_l2 = tf.get_variable(name='cnn_layer2', shape=[filter_size[1], filter_size[1], previous_channels[1], channels[1]]) #, initializer=tf.random_uniform_initializer(minval=-1, maxval=1)) #,
    bias_2 = tf.Variable(tf.constant(0.1, shape=[channels[1]], name='bias2'))

    # first input (weights)
    filter_l2 = tf.expand_dims(filter_l2, axis=0)
    all_weight_2 = tf.tile(filter_l2, [tf.shape(out_1)[0],1, 1, 1, 1])
    all_weight_2 = tf.reshape(all_weight_2, (tf.shape(out_1)[0],  channels[1] ,filter_size[1]*filter_size[1]*previous_channels[1]))

    
    # second input (images)
    all_input_2 = conv2d_inference(out_1, LAYER_2, filter_size[1], channels[1], 'conv2', 'conv2_layer', 2, channels[0], training, 12)
    all_inputs_2 = all_input_2 
  
    # stochastic layer
    with tf.variable_scope('multiplication') as scope:
      
      output_state_2    = layer_linear(all_inputs_2, (all_inputs_2.get_shape()[1], STATE_NUM), scope='layer_pre_pre')

      output_state_2    = tf.tanh(output_state_2) #elu or relu?
      output_state_2    = tf.nn.dropout(output_state_2, dropout)
      
      pre_activations_2 = layer_linear(output_state_2, (output_state_2.get_shape()[1], filter_size[1]*filter_size[1]*previous_channels[1]*channels[1]), scope='layer_pre')            
      pre_st_weight_2   = binary_wrapper(pre_activations_2, estimator =  StochasticGradientEstimator.ST, pass_through = True, stochastic_tensor = tf.constant(True), slope_tensor = slope)
      
      pre_st_weight_2   = tf.reshape(pre_st_weight_2, (tf.shape(pre_st_weight_2)[0], filter_size[1], filter_size[1], previous_channels[1], channels[1]))
      inference_fn_w    = lambda input_w: tf.multiply(filter_l2, input_w)
      st_weight_2       = tf.map_fn(inference_fn_w, pre_st_weight_2, dtype=tf.float32, swap_memory=True)
      st_weight_2       = tf.squeeze(st_weight_2, axis=1)

      input_ = (out_1, st_weight_2)
      inference_fn_out = lambda inp_: ind_2nd_multiply(inp_)
      out_2  = tf.map_fn(inference_fn_out, input_, dtype=tf.float32, swap_memory=True)
      out_2  = tf.squeeze(out_2, axis=1)
      out_2  = batch_norm(training, out_2, channels[1], 2)
      out_2  = tf.nn.relu(out_2)

  shape2_      = out_2.get_shape()
  shape_all    = int(shape2_[1])*int(shape2_[2])*channels[1]
  h_conv2_flat = tf.reshape(out_2, [-1, shape_all])

  out_3  = layer_linear(h_conv2_flat, (shape_all, LAYER_LINEAR), scope='out_linear')
  out_3  = binary_wrapper(out_3, estimator =  StochasticGradientEstimator.ST, pass_through = False, stochastic_tensor = tf.constant(True), slope_tensor = slope)
  output = layer_softmax(out_3, [LAYER_LINEAR, 10])

  loss_rf = -tf.reduce_mean(y_target * tf.log(output),reduction_indices=1)
  grad = tf.gradients(loss_rf, x_input)
    
  ###### named loss_by_example necessary for REINFORCE estimator
  tf.identity(loss_rf, name="loss_by_example")
  
  correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y_target,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

  opt = tf.train.AdamOptimizer(learning_rate_conv).minimize(loss_rf)
  
  return opt, accuracy, pre_st_weight_1, pre_activations_1, loss_rf, all_input_1, all_input_1, filter_l
  #return out_2


################################################model###################################################
iteration = 30001
validation_check = 100
layer_nodes = [100, 10]

retrain = False
mid_num = 0

# channels for each conv layer

x_input  = tf.placeholder(tf.float32, shape=[None, 784])
y_target = tf.placeholder(tf.float32, shape=[None, class_size])
training = tf.placeholder(tf.bool, name='training')
slope_l  = tf.constant(1.0)
dropout  = tf.placeholder(tf.float32) 
opt_, acc, w_st, pre_ac,  loss_, inp, conv_1, fil1 = joint_inference(x_input, training, slope_l, dropout)

############################################training start #############################################

sess = tf.Session()
sess.run(tf.initialize_all_variables())

# saver for parameters
saver = tf.train.Saver()

if retrain : 
  param_restore = directory + '/model.ckpt-' + str(mid_num)
  saver.restore(sess, param_restore)

patience = 3
patience_cnt = 0
min_delta = 0.01
hist_loss = 99.999
ealystopping_valu = 19000  


train_loss_all = []
train_acc_all  = []
for i in range(iteration):
  batch = mnist.train.next_batch(batch_size)

  _, train_loss, train_acc, weight_st, inp_, cnn1, fill = sess.run([opt_, loss_, acc, w_st, inp, conv_1, fil1], feed_dict={x_input: batch[0],  y_target: batch[1], training: True, slope_l:slope, dropout:0.5})
   
  train_loss_all.append(train_loss)
  train_acc_all.append(train_acc)

 
  ##validation checker
  if (i % validation_check) == 0 : 

    # validation
    batch_test = mnist.test.next_batch(500) 
    valid_loss, valid_acc, weight_st, inp_, cnn1 = sess.run([loss_, acc ,w_st, inp, conv_1], feed_dict={x_input: batch_test[0], y_target: batch_test[1], training: False, slope_l:slope, dropout:1.0})
    

   
    train_loss_    = np.mean(train_loss_all)
    train_accuracy = np.mean(train_acc_all)        
    test_loss      = np.mean(valid_loss)
    test_accuracy  = np.mean(valid_acc)    
   
    # save the parameters
    if not os.path.exists(directory):
      os.mkdir(directory)
    saver.save(sess, directory + '/model.ckpt', global_step=i + mid_num)    
    
    f = None
    if os.path.isfile(directory + '/log.txt'):
      f = open(directory + '/log.txt', 'a')
    else:
      f = open(directory + '/log.txt', 'w')
        
    f.write('Iteration:' + str(i+mid_num) + '\n')
    f.write('Train Loss: ' + str(train_loss_) + '\n')    
    f.write('Test Loss: ' + str(test_loss) + '\n')
    f.write('Train Accuracy: ' + str(train_accuracy) + '\n')    
    f.write('Test Accuracy: ' + str(test_accuracy) + '\n')
    
    if ((i % 1000) == 0) and (i != 0):
      if slope_annealing_rate is not None:
        slope_ = slope*slope_annealing_rate
        slope = min(slope_, 5)
        f.write("Sigmoid slope:" + str(slope) + '\n')
    
    if i >= ealystopping_valu :
      if (test_loss < hist_loss):
        patience_cnt = 0
        hist_loss = test_loss      
      else:
        patience_cnt += 1
  
      if patience_cnt > patience:
          print("early stopping...")
          break
        
    f.close()    
    # initializer of train avg
    train = []
