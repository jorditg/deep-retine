require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator
require 'string'

-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Dataset Preprocessing')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-visualize', true, 'visualize input data and weights during training')
   cmd:text()
   opt = cmd:parse(arg or {})
end

-- random seed to have reproducible results
print("Random seed used: " .. opt.seed)
torch.manualSeed(opt.seed)

-- image size
width = 512
height = 512
channels = 3
local float_size = 4
local image_size_bytes = float_size*channels*width*height

print("Image size: " .. image_size_bytes/1024 .. " kB")

-- image directory

local train_image_directory='../../3-input/train-512x512/1vsall-with-rotation/train'
local test_image_directory='../../3-input/train-512x512/1vsall-with-rotation/test'

-- CSV filename containing the label information
local labels_file='../../3-input/train-512x512/1vsall-with-rotation/labels.csv'

RGBmeans = {
   97.4545 / 255, 
   67.8075 / 255, 
   48.4029 / 255
} 
print("RGB means to use: " .. "R: " .. RGBmeans[1] .. "G: " .. RGBmeans[2] .. "B: " .. RGBmeans[3])
--global stddev of the three channels through complete dataset
RGBstddev = {
   torch.sqrt(5483.41) / 255, 
   torch.sqrt(2881.67) / 255, 
   torch.sqrt(1894.65) / 255
}
print("RGB stddev to use: " .. "R: " .. RGBstddev[1] .. "G: " .. RGBstddev[2] .. "B: " .. RGBstddev[3])

require 'DataSet'

trainData = DataSet()
testData = DataSet()
----------------------------------------------------------------------
print '==> loading dataset'
-- load all JPEG compressed images into memory
trainData:loadJPEG(train_image_directory, labels_file)
testData:loadJPEG(test_image_directory, labels_file)

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


