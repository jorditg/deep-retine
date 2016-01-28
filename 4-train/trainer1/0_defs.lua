require 'optim'

----------------------------------------------------------------------
-- GLOBAL OPTIONS
----------------------------------------------------------------------
opt = {
   seed = 1,
   save = './results/',
   batchSize = 64,
   plot = true,
   threads = 4,
   train_image_directory='../../3-input/train-256x256/1vsall/train',
   test_image_directory='../../3-input/train-256x256/1vsall/test',
   -- CSV filename containing the label information
   labels_file='../../3-input/train-256x256/1vsall/labels.csv',
   -- actual_model defines if the model is loaded (actual_model = 'load') or new defined (actual_model = 'new')
   actual_model = 'new',
   -- only if actual_mode is 'load' loads next file
   model_file = './results/model.net',
   batchCalculationType = 'parallel'
}

----------------------------------------------------------------------
-- DATASET IMAGE SIZES
----------------------------------------------------------------------
width = 256
height = 256
channels = 3

----------------------------------------------------------------------
-- MEAN AND STDDEV TO USE FOR NORMALIZATION
----------------------------------------------------------------------

RGBmeans = {
   97.4545 / 255, 
   67.8075 / 255, 
   48.4029 / 255
} 

RGBstddev = {
   74.07 / 255,
   53.71 / 255,
   43.55 / 255
}

----------------------------------------------------------------------
-- WEIGHT INITIALIZATION METHOD
----------------------------------------------------------------------
weight_init_method = 'kaiming'

----------------------------------------------------------------------
-- CLASSES NAMES
----------------------------------------------------------------------
--classes = {'1','2','3','4','0'}
classes = {'F','T'}

----------------------------------------------------------------------
-- OPTIMIZER CONFIGURATION
----------------------------------------------------------------------

-- Optimizer Adam
-- optimState = {
--  learningRate = 0.01
--}

-- Optimizer RMSProp
optimState = {
  learningRate = 0.0001
--  weightDecay = 0.0005
}
optimMethod = optim.rmsprop

-- Optimizer SGD
--optimState = {
--   learningRate = 0.0001,
--   weightDecay = 0.0,
--   momentum = 0.5,
--   dampening = 0.0,
--   learningRateDecay = 0.0,
--   nesterov = true
--}
--optimMethod = optim.sgd



