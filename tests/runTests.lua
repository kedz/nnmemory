require 'nn'
require 'rnn'
require 'memory'

nn.MemoryMaskTest()
nn.MemoryTest()
require 'cunn'
nn.MemoryTestCuda()
