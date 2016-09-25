require 'nn'
require 'cunn'

require 'memory'

local tester = torch.Tester()

local tolerance = 1e-7

local pqtests = torch.TestSuite()

function pqtests.PriorityQueueSimpleEncoderTestForwardBackward()

    local dimSize = 10
    local batchSize = 4


    local qencPlain = nn.PriorityQueueSimpleEncoder(dimSize):float()
    local qencCuda = nn.PriorityQueueSimpleEncoder(dimSize):cuda()
    qencPlain.weight:copy(qencCuda.weight)
    qencPlain.bias:copy(qencCuda.bias)
    
    local encoderSize1 = 5
    local X1Plain = torch.rand(encoderSize1, batchSize, dimSize):float()
    local X1Cuda = X1Plain:cuda()
    local gradM1 = torch.rand(encoderSize1, batchSize, dimSize):float()
    local gradPi1 = torch.rand(encoderSize1, batchSize):float()
    local gradPlain1 = {gradM1, gradPi1}
    local gradCuda1 = {gradM1:cuda(), gradPi1:cuda()}
    local outputPlain1 = qencPlain:forward(X1Plain)
    local outputCuda1 = qencCuda:forward(X1Cuda)
    local gradInputPlain = qencPlain:updateGradInput(X1Plain, gradPlain1)
    local gradInputCuda = qencCuda:updateGradInput(X1Cuda, gradCuda1)
    tester:assertTensorEq(gradInputPlain, gradInputCuda:float(), tolerance,
        "Cuda gradInput does not match cpu gradInput")
    tester:assertTensorEq(
        qencPlain.gradWeight, qencCuda.gradWeight:float(), tolerance,
        "Cuda gradWeight does not match cpu gradWeight")
    tester:assertTensorEq(
        qencPlain.gradBias, qencCuda.gradBias:float(), tolerance,
        "Cuda gradBias does not match cpu gradBias")

    local encoderSize2 = 10
    local X2Plain = torch.rand(encoderSize2, batchSize, dimSize):float()
    local X2Cuda = X2Plain:cuda()
    local gradM2 = torch.rand(encoderSize2, batchSize, dimSize):float()
    local gradPi2 = torch.rand(encoderSize2, batchSize):float()
    local gradPlain2 = {gradM2, gradPi2}
    local gradCuda2 = {gradM2:cuda(), gradPi2:cuda()}
    local outputPlain2 = qencPlain:forward(X2Plain)
    local outputCuda2 = qencCuda:forward(X2Cuda)
    local gradInputPlain = qencPlain:updateGradInput(X2Plain, gradPlain2)
    local gradInputCuda = qencCuda:updateGradInput(X2Cuda, gradCuda2)
    tester:assertTensorEq(gradInputPlain, gradInputCuda:float(), tolerance,
        "Cuda gradInput does not match cpu gradInput")
    tester:assertTensorEq(
        qencPlain.gradWeight, qencCuda.gradWeight:float(), tolerance,
        "Cuda gradWeight does not match cpu gradWeight")
    tester:assertTensorEq(
        qencPlain.gradBias, qencCuda.gradBias:float(), tolerance,
        "Cuda gradBias does not match cpu gradBias")

    local encoderSize3 = 2
    local X3Plain = torch.rand(encoderSize3, batchSize, dimSize):float()
    local X3Cuda = X3Plain:cuda()
    local gradM3 = torch.rand(encoderSize3, batchSize, dimSize):float()
    local gradPi3 = torch.rand(encoderSize3, batchSize):float()
    local gradPlain3 = {gradM3, gradPi3}
    local gradCuda3 = {gradM3:cuda(), gradPi3:cuda()}
    local outputPlain3 = qencPlain:forward(X3Plain)
    local outputCuda3 = qencCuda:forward(X3Cuda)
    local gradInputPlain = qencPlain:updateGradInput(X3Plain, gradPlain3)
    local gradInputCuda = qencCuda:updateGradInput(X3Cuda, gradCuda3)
    tester:assertTensorEq(gradInputPlain, gradInputCuda:float(), tolerance,
        "Cuda gradInput does not match cpu gradInput")
    tester:assertTensorEq(
        qencPlain.gradWeight, qencCuda.gradWeight:float(), tolerance,
        "Cuda gradWeight does not match cpu gradWeight")
    tester:assertTensorEq(
        qencPlain.gradBias, qencCuda.gradBias:float(), tolerance,
        "Cuda gradBias does not match cpu gradBias")

end


tester:add(pqtests)
tester:run()
