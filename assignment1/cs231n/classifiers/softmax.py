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
  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores = np.dot(X, W)
  # make largest value here zero for numeric stability
  scores -= np.max(scores)

  for i in xrange(num_train):
    denom = 0.0  # denominator
    for j in xrange(num_classes):
        denom += np.exp(scores[i, j])
    loss_i = -1 * scores[i, y[i]] + np.log(denom)
    loss += loss_i
    
    for j in xrange(num_classes):
        dW[:, j] += (1/denom) * np.exp(scores[i, j]) * X[i]
        if j == y[i]:
            dW[:, y[i]] -= X[i] 
  
  loss /= num_train
  dW /= num_train
  loss += 0.5 * reg * np.sum(np.square(W))
  dW += reg * W
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
  num_train = X.shape[0]
  num_classes = W.shape[1]
    
  scores = np.dot(X, W)
  scores -= np.max(scores)
  expscores = np.exp(scores)
  denoms = np.sum(expscores, axis = 1)
  
  loss = np.sum(np.log(np.sum(expscores, axis = 1)) - scores[range(num_train), y])
  
  correct_class = np.zeros((num_train, num_classes))
  correct_class[xrange(num_train), y] = -1
  dW_term1 = X.T.dot(correct_class)
  dW_term2 = (X.T / denoms).dot(expscores)

  dW = dW_term1 + dW_term2 
  
  loss /= num_train
  dW /= num_train
  
  # add in the regularization
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

