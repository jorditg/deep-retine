----------------------------------------------------------------------
--   + convolutional network (ConvNet)
--
-- It's a good idea to run this script with the interactive mode:
-- $ th -i 2_model.lua
-- this will give you a Torch interpreter at the end, that you
-- can use to play with the model.
--
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers
require 'cudnn'

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('SVHN Model Definition')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-visualize', true, 'visualize input data and weights during training')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
print '==> define parameters'

actual_model = 'load'

-- 2-class problem
noutputs = 2

----------------------------------------------------------------------
print '==> construct model'

-- find faster algorithm of cudnn
cudnn.benchmark = true

model = nn.Sequential()

if actual_model == 'load' then
  model = torch.load('./results/003/model.net')
else
--cudnn.fastest = true

-- input size 3x512x512
model = nn.Sequential()
--model:add(nn.SpatialDropout(0.2))
model:add(cudnn.SpatialConvolution(3,32, 3, 3, 1, 1))
-- 32x510x510
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(2,2,2,2))
-- 32x255x255
model:add(cudnn.SpatialConvolution(32, 32, 3, 3, 1, 1))
-- 32x255x255
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(2,2,2,2))
-- 32x126x126
model:add(cudnn.SpatialConvolution(32, 32, 3, 3, 1, 1))
-- 32x124x124
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(2,2,2,2))
-- 32x62x62
model:add(cudnn.SpatialConvolution(32, 32, 3, 3, 1, 1))
-- 32x60x60
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(2,2,2,2))
-- 32x30x30
model:add(cudnn.SpatialConvolution(32, 32, 3, 3, 1, 1))
-- 32x28x28
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(2,2,2,2))
-- 32x14x14
model:add(cudnn.SpatialConvolution(32, 48, 3, 3, 1, 1))
-- 48x12x12
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(2,2,2,2))
-- 48x6x6
model:add(nn.View(48*6*6))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(48*6*6, 512))
model:add(cudnn.ReLU(true))
model:add(nn.Linear(512, 2))
model:add(cudnn.LogSoftMax())

end
----------------------------------------------------------------------
print '==> here is the model:'
print(model)

----------------------------------------------------------------------
-- Visualization is quite easy, using itorch.image().

if opt.visualize then
      if itorch then
	 print '==> visualizing ConvNet filters'
	 print('Layer 1 filters:')
	 itorch.image(model:get(1).weight)
	 print('Layer 2 filters:')
	 itorch.image(model:get(5).weight)
      else
	 print '==> To visualize filters, start the script in itorch notebook'
      end
end

