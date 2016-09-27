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

function pqtests.LinearMemoryWriterTestGrad()
    local dimSize = DIM_SIZE
    local X = torch.rand(ENCODER_SIZE, BATCH_SIZE, dimSize)
    local mod = nn.LinearMemoryWriter(dimSize)

    local err = nn.Jacobian.testJacobian(mod, X)
    tester:assertalmosteq(
        err, 0, tolerance, 
        "LinearMemoryWriter input gradient not computed correctly.")
    local params, gradParams = mod:getParameters()

    local errParam = nn.Jacobian.testJacobianParameters(
        mod, X, params, gradParams)
    tester:assertalmosteq(
        errParam, 0, tolerance,
        "LinearMemoryWriter parameter gradient not computed correctly.")

end

tester:add(pqtests)
tester:run()
