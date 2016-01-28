require 'image'
require 'torch'


ParallelDecompressedDataSet = {}
ParallelDecompressedDataSet.__index = ParallelDecompressedDataSet

setmetatable(ParallelDecompressedDataSet, { 
    __call = function (cls, ...)
                return cls.new(...)
             end,
})

function ParallelDecompressedDataSet.new(compressed_dataset, n, c, h, w)
  local self = setmetatable({}, ParallelDecompressedDataSet)
  self.compressed = compressed_dataset
  self.channels = c
  self.batchSize = n
  self.height = h
  self.width = w
  self.data_buffers = torch.FloatTensor(2, n, c, h, w) 
  self.labels_buffers = torch.ByteTensor(2, n)
  self.threads = require 'threads'
  self.active = 1
  self.threads.Threads.serialization('threads.sharedserialize')
  self.pool = self.threads.Threads(
    1,
    function(idx)
      -- first load DataSet o be able to restore metatable in next function call
      require 'DataSet'
    end,
    function()
      compr = self.compressed
      -- trainDataTable looses his metatable inside thread. setmetatable to restore it
      -- and being able to call again its methods, later.
      setmetatable(compr, DataSet)
      decompr = self.data_buffers
      lab = self.labels_buffers
    end
  )
  return self
end

function ParallelDecompressedDataSet:launchNextDecompressionNormalization(idx, from, to, mean, stddev)
  if self.active == 1 then
    buffer_idx = 2
  else
    buffer_idx = 1
  end
  self.pool:addjob(
      function(buffer_idx)    
        compr:get_decompressed_subset(decompr:select(1, buffer_idx), idx, from, to)
        compr:get_labels_subset(lab:select(1, buffer_idx), idx, from, to)
        normalize_image_set(decompr:select(1, buffer_idx), mean, stddev)
      end,
      function() 
      end,
      buffer_idx
  )

end


function ParallelDecompressedDataSet:getNextDecompression()
  self.pool:synchronize()
  if self.active == 1 then 
     self.active = 2
  else
     self.active = 1
  end  
  self.data = self.data_buffers:select(1, self.active)
  self.labels = self.labels_buffers:select(1, self.active)
end


