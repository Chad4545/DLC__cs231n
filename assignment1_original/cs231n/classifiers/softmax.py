import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  n_sample, n_feature = X.shape # N,D
  n_class = len(np.unique(y))   # C
  
  for i in range(n_sample):
    y_i = y[i]                  # label
    
    # linear combination: score
    score_i = X[i].dot(W).reshape(1,-1) # 1,10
    
    # make softmax stable for numeric issue(-inf or inf)
    exp_score_i = np.exp(score_i - np.max(score_i,keepdims=True))
    
    # score to probability
    prob_i = exp_score_i/np.sum(exp_score_i,keepdims=1)
    
    # negative log likelihood
    loss += -np.log(prob_i[0,y_i])  
                         
    # gradient of loss w.r.t score_i
    dscore_i = prob_i.copy() #(1,10)
    dscore_i[0,y_i] -= 1
    
    # logal gradient * upstream gradient
    dW += X[i].reshape(1,-1).T.dot(dscore_i)
          # (1,3072).T.dot(1,10)  -> 3072,10
                        
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= n_sample
  reg_loss = 0.5*reg*np.sum(W*W)
  loss += reg_loss
  

  dW /= n_sample
  dW += reg*W
                         
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  n_sample, n_feature = X.shape # N,D
  n_class = len(np.unique(y))   # 10
    
  score = X.dot(W) # N,C
  exp_score = np.exp(score-np.max(score,axis=1,keepdims=1)) # N,C
  prob = exp_score/np.sum(exp_score,axis=1,keepdims=1) # N,C
  loss += np.sum(-np.log(prob[np.arange(n_sample),y])) # scalar
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= n_sample
  reg_loss = 0.5*reg*np.sum(W*W)
  loss += reg_loss

  #upstream gradient  
  dscore = prob.copy()
  dscore[np.arange(n_sample),y] -= 1
  dscore /= n_sample
  
  # Gradient of loss w.r.t weights
  dW = X.T.dot(dscore) # D,C
  dW += reg*W
  return loss, dW

