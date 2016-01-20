----------------------------------------------------------------------
-- Jordi de la Torre
----------------------------------------------------------------------
require 'torch'

----------------------------------------------------------------------
print '==> processing options'

cmd = torch.CmdLine()
cmd:text()
cmd:text('SVHN Loss Function')
cmd:text()
cmd:text('Options:')
-- global:
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 6, 'number of threads')
-- training:
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:option('-plot', true, 'live plot')
cmd:option('-learningRate', 0.01, 'learning rate at t=0')
cmd:option('-batchSize', 64, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0.0005, 'weight decay')
cmd:option('-momentum', 0.9, 'momentum')
cmd:option('-dampening', 0, 'momentum dampening')
cmd:option('-learningRateDecay', 0.0, 'learningRateDecay')
cmd:option('-nesterov', true, 'Enables Nesterov Acceleration Gradient')
cmd:text()
opt = cmd:parse(arg or {})

-- nb of threads and fixed seed (for repeatable experiments)
print('==> switching to CUDA')
require 'cunn'
torch.setdefaulttensortype('torch.FloatTensor')

torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

----------------------------------------------------------------------
print '==> executing all'

dofile '1_data.lua'
dofile '2_model.lua'
dofile '3_loss.lua'
dofile '4_train.lua'
dofile '5_test.lua'

----------------------------------------------------------------------
print '==> training!'

while true do
   train()
   collectgarbage()
   collectgarbage()
   test()
   collectgarbage()
   collectgarbage()
end

