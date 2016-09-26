require 'nn'
require 'memory'

dimSize = 200
batchSize = 100
encoderSize = 35
decoderSize = 35

numTests = 100

local X = torch.rand(encoderSize, batchSize, dimSize)
local Pi = torch.exp(torch.rand(encoderSize, batchSize))
Pi = torch.cdiv(Pi, Pi:sum(1):expand(encoderSize, batchSize))
Pi, I = torch.sort(Pi, 1, true)
local Y = torch.rand(decoderSize, batchSize, dimSize)
local gradY = torch.rand(decoderSize, batchSize, dimSize)

input = {X, Pi, Y}

forwardTimes = {}
backwardTimes = {}



qdec = nn.PriorityQueueSimpleDecoder(dimSize)
for t=1,numTests do
    local startForward = os.clock()
    local output = qdec:forward(input)
    local stopForward = os.clock()
    table.insert(forwardTimes, stopForward - startForward)
    local startBackward = os.clock()
    qdec:backward(input, gradY)
    local stopBackward = os.clock()
    table.insert(backwardTimes, stopBackward - startBackward)
end
print("nn.PriorityQueueSimpleDecoder")
print("Avg forward time = ".. torch.Tensor(forwardTimes):mean())
print("Avg backward time = ".. torch.Tensor(backwardTimes):mean())
