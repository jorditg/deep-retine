----------------------------------------------------------------------
-- This script demonstrates how to define a couple of different
-- loss functions:
--   + negative-log likelihood, using log-normalized output units (SoftMax)
----------------------------------------------------------------------

require 'torch'   -- torch
require 'nn'      -- provides all sorts of loss functions

print '==> define loss'

-- This loss requires the outputs of the trainable model to
-- be properly normalized log-probabilities, which can be
-- achieved using a softmax function as the last layer of the model 
-- (check that done in this way!!)

-- The loss works like the MultiMarginCriterion: it takes
-- a vector of classes, and the index of the grountruth class
-- as arguments.

criterion = nn.ClassNLLCriterion()

----------------------------------------------------------------------
print '==> here is the loss function:'
print(criterion)

