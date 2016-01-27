require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator
require 'string'

-- random seed to have reproducible results
print("Random seed used: " .. opt.seed)
torch.manualSeed(opt.seed)

local float_size = 4
local image_size_bytes = float_size*channels*width*height

print("Image size: " .. image_size_bytes/1024 .. " kB")

require 'DataSet'

trainData = DataSet()
testData = DataSet()
----------------------------------------------------------------------
print '==> loading dataset'
-- load all JPEG compressed images into memory
trainData:loadJPEG(opt.train_image_directory, opt.labels_file)
testData:loadJPEG(opt.test_image_directory, opt.labels_file)

----------------------------------------------------------------------
-- training/test size

-- trsize is the number of files multiplied by 4 because every image is present in the
-- dataset with no transformation, horizontal flipped, vertical flipped, and both.
trsize = trainData:size()
tesize = testData:size()
----------------------------------------------------------------------

-- Converting original dataset to a 1vsall
for i = 1,trsize do
  if trainData.labels[i] == 0 then
    trainData.labels[i] = 1
  else
    trainData.labels[i] = 2
  end
end

for i = 1,tesize do
  if testData.labels[i] == 0 then
    testData.labels[i] = 1
  else
    testData.labels[i] = 2
  end
end


