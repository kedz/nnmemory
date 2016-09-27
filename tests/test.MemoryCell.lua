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

function pqtests.MemoryCellTestGrad()
    local dimSize = DIM_SIZE
    local X = torch.rand(ENCODER_SIZE, BATCH_SIZE, dimSize)
    local cell = nn.MemoryCell():add(nn.LinearMemoryWriter(dimSize)):add(
        nn.BilinearAttentionMemoryWriter(dimSize))

    local join = nn.Sequential():add(cell):add(nn.SortOnKey(true)):add(
    nn.ParallelTable():add(nn.Identity()):add(nn.Sequential():add(
        nn.Replicate(dimSize, 3, 2)))
    ):add(nn.CMulTable()):add(nn.Sum(1,3,false))

    local err = nn.Jacobian.testJacobian(join, X)
    tester:assertalmosteq(
        err, 0, tolerance, 
        "MemoryCell input gradient not computed correctly.")

    local params, gradParams = join:getParameters()

    local errParam = nn.Jacobian.testJacobianParameters(
        join, X, params, gradParams)
    tester:assertalmosteq(
        errParam, 0, tolerance,
        "MemoryCell parameter gradient not computed correctly.")
end

function pqtests.MemoryCellTestMaskZero()
    local batchSize = 2
    local dimSize = DIM_SIZE
    local maxSteps = 3

    local X = torch.rand(maxSteps, batchSize, dimSize)
    local Y = torch.rand(2, batchSize, dimSize)

    X[2][1]:fill(0)
    X[3][2]:fill(0)
    Y[1]:copy(X[1])
    Y[2][1] = X[3][1]
    Y[2][2] = X[2][2]


    local cell = nn.MemoryCell():add(nn.LinearMemoryWriter(dimSize)):add(
        nn.LinearMemoryWriter(dimSize))
    cell:maskZero()

    local join = nn.Sequential():add(cell):add(nn.SortOnKey(true)):add(
    nn.ParallelTable():add(nn.Identity()):add(nn.Sequential():add(
        nn.Replicate(dimSize, 3, 2)))
    ):add(nn.CMulTable()):add(nn.Sum(1,3,false))

   
    local G = torch.rand(batchSize, dimSize)

    local params, gradParams = join:getParameters()

    join:zeroGradParameters()
    local outputX = join:forward(X)
    local gradX = join:backward(X, G):clone()
    local gradXparams = gradParams:clone()

    join:zeroGradParameters()
    local outputY = join:forward(Y)
    local gradY = join:backward(Y, G):clone()
    local gradYparams = gradParams:clone()

    tester:asserteq(outputX, outputY)
    tester:assertTensorEq(gradX[1], gradY[1])
    tester:assertTensorEq(gradX[3][1], gradY[2][1])
    tester:assertTensorEq(gradX[2][1], torch.zeros(dimSize))
    tester:assertTensorEq(gradX[3][2], torch.zeros(dimSize))
    tester:assertTensorEq(gradX[2][2], gradY[2][2])
    tester:assertTensorEq(gradXparams, gradYparams)

end

tester:add(pqtests)
tester:run()
