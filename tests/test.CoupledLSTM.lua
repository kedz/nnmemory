require 'nn'
require 'rnn'
require 'memory'

local tester = torch.Tester()

local tolerance = .000000001
local DIM_SIZE = 5
local BATCH_SIZE = 2
local ENCODER_SIZE = 4
local DECODER_SIZE = 4


local pqtests = torch.TestSuite()

function pqtests.CoupledLSTMTestGrad()
    local numLayers = 2
    local input = torch.rand(2, ENCODER_SIZE, BATCH_SIZE, DIM_SIZE)

    local mod = nn.CoupledLSTM(DIM_SIZE, numLayers)

    local net = nn.Sequential():add(nn.SplitTable(1, 4)):add(mod):add(
        nn.JoinTable(1,3))

    local err = nn.Jacobian.testJacobian(net, input)
    tester:assertalmosteq(err, 0, tolerance, 
        "CoupledLSTM input gradient is incorrect.")
    local params, gradParams = mod:getParameters()

    net:zeroGradParameters()
    local errParam = nn.Jacobian.testJacobianParameters(
        net, input, params, gradParams)
    tester:assertalmosteq(
        errParam, 0, tolerance,
        "CoupledLSTM parameter gradient not computed correctly.")

    mod:decouple()
    net:zeroGradParameters()
    local err = nn.Jacobian.testJacobian(net, input)
    tester:assertalmosteq(err, 0, tolerance, 
        "CoupledLSTM input gradient is incorrect.")
    local params, gradParams = mod:getParameters()

    net:zeroGradParameters()
    local errParam = nn.Jacobian.testJacobianParameters(
        net, input, params, gradParams)
    tester:assertalmosteq(
        errParam, 0, tolerance,
        "CoupledLSTM parameter gradient not computed correctly.")
       
end

tester:add(pqtests)
tester:run()
