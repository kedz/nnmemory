require 'nn'
require 'memory'

dimSize = 200
batchSize = 100
encoderSize = 35
decoderSize = 35

numTests = 100

local X = torch.rand(encoderSize, batchSize, dimSize)
local gradM = torch.rand(encoderSize, batchSize, dimSize)
local gradpi = torch.rand(encoderSize, batchSize)

forwardTimes = {}
backwardTimes = {}

qenc = nn.PriorityQueueSimpleEncoder(dimSize)
for t=1,numTests do
    local startForward = os.clock()
    local output = qenc:forward(X)
    local stopForward = os.clock()
    table.insert(forwardTimes, stopForward - startForward)
    local startBackward = os.clock()
    qenc:backward(X, {gradM, gradpi})
    local stopBackward = os.clock()
    table.insert(backwardTimes, stopBackward - startBackward)
end
print("nn.PriorityQueueSimpleEncoder")
print("Avg forward time = ".. torch.Tensor(forwardTimes):mean())
print("Avg backward time = ".. torch.Tensor(backwardTimes):mean())
