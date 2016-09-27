require 'nn'
require 'rnn'
require 'memory'

local tester = torch.Tester()

local tolerance = .000000001
local DIM_SIZE = 5
local BATCH_SIZE = 2
local ENCODER_SIZE = 3
local DECODER_SIZE = 4


local pqtests = torch.TestSuite()

function pqtests.BilinearAttentionMemoryWriterTestGrad()
    local X = torch.rand(ENCODER_SIZE, BATCH_SIZE, DIM_SIZE)
    local mod = nn.BilinearAttentionMemoryWriter(DIM_SIZE)

    local err = nn.Jacobian.testJacobian(mod, X)
    tester:assertalmosteq(err, 0, tolerance, 
        "BilinearAttentionMemoryWriter input gradient is incorrect.")
    local params, gradParams = mod:getParameters()

end

tester:add(pqtests)
tester:run()
