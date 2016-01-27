require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods


-- switch to CUDA
model:cuda()
criterion:cuda()

----------------------------------------------------------------------
print '==> defining some tools'

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- Log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
lossLogger = optim.Logger(paths.concat(opt.save, 'loss.log'))

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
if model then
   parameters,gradParameters = model:getParameters()
end


-- Allocate memory for decompressed data
require 'DecompressedDataSet'
train_data = DecompressedDataSet(trainData, opt.batchSize, channels, height, width)

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
   local current_loss = 0.0
   for t = 1,trsize,opt.batchSize do
      -- disp progress
      xlua.progress(t, trsize)

      -- create mini batch
      local inputs = {}
      local targets = {}

      local from = t
      local to = math.min(t+opt.batchSize-1,trsize)
      --time_mini = sys.clock()

      local rows = to - from + 1

      train_data:decompress(shuffle, from, to)
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
      _, fs = optimMethod(feval, parameters, optimState)
      current_loss = current_loss + fs[1]
   end
   current_loss = current_loss / (trsize / opt.batchSize)
   print('==> current loss = ' .. current_loss)
   print ''
   -- update logger/plot
   lossLogger:add{['mean loss (train set)'] = current_loss}
   if opt.plot then
      lossLogger:style{['mean loss (train set)'] = '-'}
      lossLogger:plot()
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

