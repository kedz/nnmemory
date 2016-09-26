require 'nn'
require 'cunn'
require 'memory'

local tester = torch.Tester()
local tolerance = 1e-7
local pqtests = torch.TestSuite()
local DIM_SIZE = 2
local BATCH_SIZE = 2
local ENCODER_SIZE = 3
local DECODER_SIZE = 4


function randomData(encoderSize, decoderSize, batchSize, dimSize)
    local X = torch.rand(encoderSize, batchSize, dimSize)
    local Y = torch.rand(decoderSize, batchSize, dimSize)
    local pi = torch.rand(encoderSize, batchSize)
    local Z = torch.exp(pi, pi):sum(1):expand(encoderSize, batchSize)
    torch.cdiv(pi, pi, Z)
    pi, _ = torch.sort(pi, 1, true)

    return {X, pi, Y}
    
end

function pqtests.PriorityQueueSimpleDecoderTestForwardBackward()
    local dimSize = DIM_SIZE
    local batchSize = BATCH_SIZE
    local encoderSize = ENCODER_SIZE
    local decoderSize = DECODER_SIZE
    local input = randomData(encoderSize, decoderSize, batchSize, dimSize)
    local X, pi, Y = unpack(input) 

    local qdecPlain = nn.PriorityQueueSimpleDecoder(dimSize):float()
    local qdecCuda = nn.PriorityQueueSimpleDecoder(dimSize):cuda()
    qdecPlain.weight_read_in:copy(qdecCuda.weight_read_in)
    qdecPlain.weight_read_h:copy(qdecCuda.weight_read_h)
    qdecPlain.weight_read_b:copy(qdecCuda.weight_read_b)
    qdecPlain.weight_forget_in:copy(qdecCuda.weight_forget_in)
    qdecPlain.weight_forget_h:copy(qdecCuda.weight_forget_h)
    qdecPlain.weight_forget_b:copy(qdecCuda.weight_forget_b)
    local inputPlain = {X:float(), pi:float(), Y:float()}
    local inputCuda = {X:cuda(), pi:cuda(), Y:cuda()}
    local outputPlain = qdecPlain:forward(inputPlain)
    local outputCuda = qdecCuda:forward(inputCuda)
    local GPlain = torch.rand(decoderSize, batchSize, dimSize):float()
    local GCuda = GPlain:cuda()
    local gradInputPlain = qdecPlain:updateGradInput(inputPlain, GPlain)
    local gradInputCuda = qdecCuda:updateGradInput(inputCuda, GCuda)

    tester:assertTensorEq(outputPlain, outputCuda:float(), 
        tolerance,
        "Cuda output does not match cpu output")
    tester:assertTensorEq(gradInputPlain[1], gradInputCuda[1]:float(), 
        tolerance,
        "Cuda gradInput[1] does not match cpu gradInput[1]")
    tester:assertTensorEq(gradInputPlain[2], gradInputCuda[2]:float(), 
        tolerance,
        "Cuda gradInput[2] does not match cpu gradInput[2]")
    tester:assertTensorEq(gradInputPlain[3], gradInputCuda[3]:float(), 
        tolerance,
        "Cuda gradInput[3] does not match cpu gradInput[3]")

    tester:assertTensorEq(
        qdecPlain.grad_read_in,
        qdecCuda.grad_read_in:float(), 
        tolerance,
        "Cuda grad_read_in does not match cpu grad_read_in")
    tester:assertTensorEq(
        qdecPlain.grad_read_h,
        qdecCuda.grad_read_h:float(), 
        tolerance,
        "Cuda grad_read_h does not match cpu grad_read_h")
    tester:assertTensorEq(
        qdecPlain.grad_read_b,
        qdecCuda.grad_read_b:float(), 
        tolerance,
        "Cuda grad_read_b does not match cpu grad_read_b")

    tester:assertTensorEq(
        qdecPlain.grad_forget_in,
        qdecCuda.grad_forget_in:float(), 
        tolerance,
        "Cuda grad_forget_in does not match cpu grad_forget_in")
    tester:assertTensorEq(
        qdecPlain.grad_forget_h,
        qdecCuda.grad_forget_h:float(), 
        tolerance,
        "Cuda grad_forget_h does not match cpu grad_forget_h")
    tester:assertTensorEq(
        qdecPlain.grad_forget_b,
        qdecCuda.grad_forget_b:float(), 
        tolerance,
        "Cuda grad_forget_b does not match cpu grad_forget_b")




end

tester:add(pqtests)
tester:run()
