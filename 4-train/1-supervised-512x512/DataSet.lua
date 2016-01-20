require 'image'
require 'torch'


DataSet = {}
DataSet.__index = DataSet

setmetatable(DataSet, { 
    __call = function (cls, ...)
                return cls.new(...)
             end,
})

function DataSet.new()
  local self = setmetatable({}, DataSet)
  self.image_extension = '.jpeg'
  self.imagesSet = {}
  self.fileList = {}
  --self.labels = nil
  return self
end

function DataSet:size()
    return table.getn(self.imagesSet)
end

-- Loads the CSV file of image name, labels into a lua table
function DataSet:loadLabels(labelsCSVfile)
    local labels = {}
    -- Load as a table the labels of each image
    for line in io.lines(labelsCSVfile) do
        local _, _, name, label = string.find(line, "(.*),(%d*)")
        labels[name .. self.image_extension ] = label
    end
    return labels
end

-- Returns the file list of files inside a directory that have the image extension
-- indicated as a paramater
function DataSet:listFiles(image_directory, image_extension)
  function image_extension_end (filename)
    if string.find(filename, self.image_extension .. "$") then
      return true
    end
    return false
  end
  fl = {}
  i = 1
  for f in paths.files(image_directory, image_extension_end) do
    fl[i] = f
    i = i + 1
  end
  return fl
end

-- Loads into memory all the JPEG compressed files of directory 'dir'
-- The required RAM memory is the same as the disk size
function DataSet:loadJPEG(dir, labelsCSVFile)
    self.fileList = self:listFiles(dir, image_extension)
    local number_files = table.getn(self.fileList)
    self.labels = torch.ByteTensor(number_files)
    local CSVLabels = self:loadLabels(labelsCSVFile)
    self.imagesSet = {}
    for i = 1, number_files do
        local fin = torch.DiskFile(dir .. "/" .. self.fileList[i], 'r')
        fin:binary()
        fin:seekEnd()
        local file_size_bytes = fin:position() - 1
        fin:seek(1)
        self.imagesSet[i] = torch.ByteTensor(file_size_bytes)
        fin:readByte(self.imagesSet[i]:storage())
        fin:close()
        -- Find labels for each file
        self.labels[i] = CSVLabels[self.fileList[i]]
    end
end

-- Decompresses a subset of the dataset referenced by the 'idx' vector
function DataSet:get_decompressed_subset(idx, channels, height, width, from, to)
    -- 'from' and 'to' are indexes from vector idx for using a subset of idx
    -- default behaviour: selecting all indexes
    from = from or 1
    to = to or idx:size()[1]
    local rows = to - from + 1
    local data_v = torch.FloatTensor(rows, channels, height, width)
    local target_v = torch.ByteTensor(rows)
    local n = table.getn(self.imagesSet)
    for i = from, to do
        img_binary = self.imagesSet[idx[i]]
        im = image.decompressJPG(img_binary)
        local j = i - from + 1
        data_v[{j, {}, {}, {}}] = im
        target_v[j] = self.labels[idx[i]]
    end
    return {
        data = data_v,
        labels = target_v
    }
end

function DataSet:get_data_image(i)
    img_binary = self.imagesSet[i]
    im = image.decompressJPG(img_binary)
    channels = im:size()[1]
    return im
end

function normalize_image_set(decompressed_image_set, RGBm, RGBs)
    local n = decompressed_image_set:size()[1]
    local channels = decompressed_image_set:size()[2]
    for i = 1, n do
        for c = 1,channels do
            decompressed_image_set[{i, c, {}, {}}]:add(-RGBm[c])
            decompressed_image_set[{i, c, {}, {}}]:div(RGBs[c])
        end
    end
end

function normalize_image(decompressed_image, RGBm, RGBs)
    local channels = decompressed_image:size()[1]
    for c = 1,channels do
        decompressed_image[{c, {}, {}}]:add(-RGBm[c])
        decompressed_image[{c, {}, {}}]:div(RGBs[c])
    end
end

