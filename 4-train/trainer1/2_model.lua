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
-- input size 3x256x256
model = nn.Sequential()
model:add(nn.SpatialConvolution(3, 40, 5, 5, 1, 1))
--  40x252x252
model:add(nn.ReLU(true))
model:add(nn.SpatialMaxPooling(2,2,2,2))
-- 40x126x126
model:add(nn.SpatialConvolution(40, 20, 9, 9, 1, 1))
-- 20x118x118
model:add(nn.ReLU(true))
model:add(nn.SpatialMaxPooling(2,2,2,2))
-- 20x59x59
model:add(nn.SpatialConvolution(20, 10, 12, 12, 1, 1))
-- 10x48x48
model:add(nn.ReLU(true))
model:add(nn.SpatialMaxPooling(2,2,2,2))
-- 10x24x24
model:add(nn.ReLU(true))
model:add(nn.SpatialMaxPooling(2,2,2,2))
-- 10x12x12 = 1440
model:add(nn.View(1440))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(1440, 512))
model:add(nn.ReLU(true))
model:add(nn.Linear(512, 2))

print '==> define loss'
model:add(nn.LogSoftMax())

-- weight initialization
-- Must be done in nn (not implemented for cuDNN, conversion after)
print ('==> weight initialization. Method =>' .. weight_init_method)
model = require('weight-init')(model, weight_init_method)

-- conversion to cuDNN
print('==> Conversion from nn to cuDNN')
cudnn.convert(model, cudnn)

end

----------------------------------------------------------------------
print '==> here is the model:'
print(model)

criterion = nn.ClassNLLCriterion()

print '==> here is the loss function:'
print(criterion)
