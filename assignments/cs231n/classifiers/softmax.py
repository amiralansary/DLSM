import numpy as np
from random import shuffle

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
  dW   = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # compute the loss and the gradient
  num_classes           = W.shape[1]
  num_train             = X.shape[0]
  
  for i in xrange(num_train):

    scores              = X[i].dot(W)
    scores_norm         = scores - np.max(scores)    # normalizaion trick
    scores_norm_exp     = np.exp(scores_norm)
    probs               = scores_norm_exp/np.sum(scores_norm_exp)
    prob_correct_log    = -np.log(probs[y[i]])

    loss                += prob_correct_log

    dscores             = probs.copy()
    dscores[y[i]]       -=1
    
    for j in xrange(num_classes):
        dW[:,j]     += X[i]*dscores[j]
        
        
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss  /= num_train
  dW    /= num_train
  # Add regularization to the loss.
  loss  += 0.5 * reg * np.sum(W * W)
  dW    += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  # compute the loss and the gradient
  num_classes       = W.shape[1]
  num_train         = X.shape[0]
  
  scores            = X.dot(W)
  scores_norm       = scores - np.max(scores)
  scores_norm_exp   = np.exp(scores_norm)
  probs             = scores_norm_exp / np.sum(scores_norm_exp, axis=1, keepdims=True)
  probs_correct_log = -np.log(probs[xrange(num_train),y])
  
  loss = np.sum(probs_correct_log)/num_train
  loss += 0.5*reg*np.sum(W*W)

  dscores = probs.copy()
  dscores[range(num_train),y] -= 1
  dscores /= num_train
  
  dW = np.dot(X.T, dscores)
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

