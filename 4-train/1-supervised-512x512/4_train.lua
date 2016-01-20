----------------------------------------------------------------------
-- This script demonstrates how to define a training procedure,
-- irrespective of the model/loss functions chosen.
--
-- It shows how to:
--   + construct mini-batches on the fly
--   + define a closure to estimate (a noisy) loss
--     function, as well as its derivatives wrt the parameters of the
--     model to be trained
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Training/Optimization')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
   cmd:option('-visualize', false, 'visualize input data and weights during training')
   cmd:option('-plot', true, 'live plot')
   cmd:option('-learningRate',0.1, 'learning rate at t=0')
   cmd:option('-batchSize', 64, 'mini-batch size (1 = pure stochastic)')
   cmd:option('-weightDecay', 0, 'weight decay')
   cmd:option('-momentum', 0.5, 'momentum')
   cmd:option('-dampening', 0, 'momentum dampening')
   cmd:option('-learningRateDecay', 1e-7, 'learningRateDecay')
   cmd:option('-nesterov', false, 'Enables Nesterov Acceleration Gradient')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------

model:cuda()
criterion:cuda()

----------------------------------------------------------------------
print '==> defining some tools'

-- classes
--classes = {'1','2','3','4','0'}
classes = {'F','T'}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- Log results to files
hyperparamsLogger = optim.Logger(paths.concat(opt.save, 'hyperparams.log'))
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))


hyperparamsLogger:add{['learningRate'] = opt.learningRate,
                      ['learningRateDecay'] = opt.learningRateDecay,
                      ['batchSize'] = opt.batchSize,
                      ['weightDecay'] = opt.weightDecay,
                      ['momentum'] = opt.momentum,
                      ['dampening'] = opt.dampening,
                      ['nesterov'] = tostring(opt.nesterov)}

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
if model then
   parameters,gradParameters = model:getParameters()
end

----------------------------------------------------------------------
print '==> configuring optimizer'

optimState = {
   learningRate = opt.learningRate,
   weightDecay = opt.weightDecay,
   momentum = opt.momentum,
   dampening = opt.dampening,
   learningRateDecay = opt.learningRateDecay,
   nesterov = opt.nesterov
}
optimMethod = optim.sgd

----------------------------------------------------------------------
print '==> defining training procedure'

function train()

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()

   -- shuffle at each epoch
   shuffle = torch.randperm(trsize)

   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,trsize,opt.batchSize do
      -- disp progress
      xlua.progress(t, trsize)

      -- create mini batch
      local inputs = {}
      local targets = {}

      local from = t
      local to = math.min(t+opt.batchSize-1,trsize)
      --time_mini = sys.clock()
      local train_data = trainData:get_decompressed_subset(shuffle, channels, height, width, from, to)
      normalize_image_set(train_data.data, RGBmeans, RGBstddev)
      --local time_mini = sys.clock() - time_mini
      --print("Time to decompress one sample" .. (time_mini/opt.batchSize*1000) .. "ms")
      local actualBatchSize = to - from + 1
      for i = 1,actualBatchSize do
         -- load new sample
         local input = train_data.data[{i, {},{},{}}]
         local target = train_data.labels[i]
         input = input:cuda()
         table.insert(inputs, input)
         table.insert(targets, target)
      end
      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
                       -- get new parameters
                       if x ~= parameters then
                          parameters:copy(x)
                       end

                       -- reset gradients
                       gradParameters:zero()

                       -- f is the average of all criterions
                       local f = 0

                       -- evaluate function for complete mini batch
                       for i = 1,#inputs do
                          -- estimate f
                          local output = model:forward(inputs[i])
                          local err = criterion:forward(output, targets[i])
                          f = f + err

                          -- estimate df/dW
                          local df_do = criterion:backward(output, targets[i])
                          model:backward(inputs[i], df_do)

                          -- update confusion
                          confusion:add(output, targets[i])
                       end

                       -- normalize gradients and f(X)
                       gradParameters:div(#inputs)
                       f = f/#inputs

                       -- return f and df/dX
                       return f,gradParameters
                    end

      -- optimize on current mini-batch
      optimMethod(feval, parameters, optimState)
   end

   -- time taken
   time = sys.clock() - time
   time = time / trsize
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update logger/plot
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   if opt.plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      trainLogger:plot()
   end

   -- save/log current net
   local filename = paths.concat(opt.save, 'model.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)

   -- next epoch
   confusion:zero()
   epoch = epoch + 1
end

