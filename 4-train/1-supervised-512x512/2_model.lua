require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers
require 'cudnn'

----------------------------------------------------------------------
print '==> define parameters'


----------------------------------------------------------------------
print '==> construct model'

-- find faster algorithm of cudnn
cudnn.benchmark = true

model = nn.Sequential()

if opt.actual_model == 'load' then
  model = torch.load(opt.model_file)
else
-- input size 3x512x512
model = nn.Sequential()
model:add(cudnn.SpatialConvolution(3, 32, 5, 5, 1, 1))
-- 32x508x508
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(3,3,3,3))
-- 32x169x169
model:add(cudnn.SpatialConvolution(32, 32, 8, 8, 1, 1))
-- 32x162x162
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(3,3,3,3))
-- 32x54x54
model:add(cudnn.SpatialConvolution(32, 32, 16, 16, 1, 1))
-- 32x39x39
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(3,3,3,3))
-- 32x13x13
model:add(cudnn.SpatialConvolution(32, 64, 10, 10, 1, 1))
-- 64x4x4
model:add(cudnn.ReLU(true))
model:add(nn.View(1024))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(1024, 512))
model:add(cudnn.ReLU(true))
model:add(nn.Linear(512, 2))

print '==> define loss'
model:add(cudnn.LogSoftMax())

-- weight initialization
-- Must be done in nn (not implemented for cuDNN, conversion after)
print ('==> weight initialization. Method =>' .. weight_init_method)
model = require('weight-init')(model, weight_init_method)

-- conversion to cuDNN
--print('==> Conversion from nn to cuDNN')
--cudnn.convert(model, cudnn)

end

----------------------------------------------------------------------
print '==> here is the model:'
print(model)

criterion = nn.ClassNLLCriterion()

print '==> here is the loss function:'
print(criterion)
