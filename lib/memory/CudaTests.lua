local cumemtest = torch.TestSuite()
local precision_forward = 1e-4
local precision_backward = 1e-2

local mytester = torch.Tester()

local function makeCudaInput(input)
    if type(input) == 'table' then
        local cudaInput = {}
        for k,v in ipairs(input) do
            table.insert(cudaInput, makeCudaInput(v))
        end
        return cudaInput
    else
        return input:cuda()
    end
end

local function makeGrad(output)
    if type(output) == 'table' then
        local grad = {}
        for k,v in ipairs(output) do
            table.insert(grad, makeGrad(v))
        end
        return grad
    else
        return output:clone():zero():add(torch.rand(output:nElement()))
    end
end


local function testForward(protoModule, input, maxError)
    local groundTruth = protoModule:forward(input)
    local cudaMod = protoModule:clone():cuda()
    local cudaInput = makeCudaInput(input)
    local cudaForward = cudaMod:forward(cudaInput)
    if type(groundTruth) == "table" then
        for i,output in ipairs(groundTruth) do
            local error = cudaForward[i]:float() - output
            mytester:assertlt(error:abs():max(), maxError, 
                'error on state (forward) ')
        end
    else
        local error = cudaForward:float() - groundTruth
        mytester:assertlt(error:abs():max(), maxError, 
            'error on state (forward) ')
    end
end

local function testBackward(protoModule, input, maxError)
    protoModule:zeroGradParameters()
    local cudaMod = protoModule:clone():cuda()

    local params, gradParams = protoModule:getParameters()
    local output = protoModule:forward(input)
    local outputGrad = makeGrad(output)
    local inputGrad = protoModule:backward(input, outputGrad)   
 
    cudaMod:zeroGradParameters()
    local cudaParams, cudaGradParams = cudaMod:getParameters()
    local cudaInput = makeCudaInput(input)
    local cudaOutput = cudaMod:forward(cudaInput)
    local cudaOutputGrad = makeCudaInput(outputGrad)
    local cudaInputGrad = cudaMod:backward(cudaInput, cudaOutputGrad)

    if type(inputGrad) == "table" then
        for i,grad in ipairs(inputGrad) do
            local error = cudaInputGrad[i]:float() - grad
            mytester:assertlt(error:abs():max(), maxError, 
                'error on state (backward) ')
        end
    else
        local error = cudaInputGrad:float() - inputGrad
        mytester:assertlt(error:abs():max(), maxError, 
            'error on state (backward) ')
    end

    if gradParams:dim() > 0 then
        local errorParams = cudaGradParams:float() - gradParams
        mytester:assertlt(errorParams:abs():max(), maxError, 
            'error on parameters (backward) ')
    end
end


function cumemtest.PriorityQueueSimpleDecoderV2()
    local dimSize = 10
    local batchSize = 3
    local encoderSize = 4
    local decoderSize = 5
    local M = torch.rand(encoderSize, batchSize, dimSize)
    local Y = torch.rand(decoderSize, batchSize, dimSize)
    local pi = torch.rand(encoderSize, batchSize)
    local Z = torch.exp(pi, pi):sum(1):expand(encoderSize, batchSize)
    torch.cdiv(pi, pi, Z)
    pi, _ = torch.sort(pi, 1, true)

    local qdec = nn.PriorityQueueSimpleDecoderV2(dimSize)
    testForward(qdec, {M, pi, Y}, precision_forward)
    qdec.prevQueueSize = 0
    testBackward(qdec, {M, pi, Y}, precision_backward)

end


function cumemtest.PriorityQueueSimpleDecoder()
    local dimSize = 10
    local batchSize = 3
    local encoderSize = 4
    local decoderSize = 5
    local M = torch.rand(encoderSize, batchSize, dimSize)
    local Y = torch.rand(decoderSize, batchSize, dimSize)
    local pi = torch.rand(encoderSize, batchSize)
    local Z = torch.exp(pi, pi):sum(1):expand(encoderSize, batchSize)
    torch.cdiv(pi, pi, Z)
    pi, _ = torch.sort(pi, 1, true)

    local qdec = nn.PriorityQueueSimpleDecoder(dimSize)
    testForward(qdec, {M, pi, Y}, precision_forward)
    testBackward(qdec, {M, pi, Y}, precision_backward)

end

function cumemtest.LinearAssociativeMemoryWriterP()
    local encoderSize = 5
    local dimSize = 10
    local batchSize = 3

    local X = torch.rand(encoderSize, batchSize, dimSize)

    local netF = nn.LinearAssociativeMemoryWriterP(dimSize, "forward")
    testForward(netF, X, precision_forward)
    testBackward(netF, X, precision_backward)

    local netB = nn.LinearAssociativeMemoryWriterP(dimSize, "backward")
    testForward(netB, X, precision_forward)
    testBackward(netB, X, precision_backward)

    local netA = nn.LinearAssociativeMemoryWriterP(dimSize, "all")
    testForward(netA, X, precision_forward)
    testBackward(netA, X, precision_backward)

end



function cumemtest.LinearAssociativeMemoryWriter()
    local encoderSize = 5
    local dimSize = 10
    local batchSize = 3

    local X = torch.rand(encoderSize, batchSize, dimSize)

    local netF = nn.LinearAssociativeMemoryWriter(dimSize, "forward")
    testForward(netF, X, precision_forward)
    testBackward(netF, X, precision_backward)

    local netB = nn.LinearAssociativeMemoryWriter(dimSize, "backward")
    testForward(netB, X, precision_forward)
    testBackward(netB, X, precision_backward)

    local netA = nn.LinearAssociativeMemoryWriter(dimSize, "all")
    testForward(netA, X, precision_forward)
    testBackward(netA, X, precision_backward)

end



function cumemtest.LinearAssociativeMemoryReader()
    local encoderSize = 5
    local decoderSize = 6
    local dimSize = 10
    local batchSize = 3

    local M = torch.rand(encoderSize, batchSize, dimSize)
    local Y = torch.rand(decoderSize, batchSize, dimSize)

    local mod = nn.LinearAssociativeMemoryReader(dimSize)
    testForward(mod, {M, Y}, precision_forward)
    testBackward(mod, {M, Y}, precision_forward)

end

function cumemtest.SortOnKey()
    local encoderSize = 4
    local batchSize = 2
    local dimSize = 10
 
    local M = torch.rand(encoderSize, batchSize, dimSize)
    local pi = torch.rand(encoderSize, batchSize)

    local mod = nn.SortOnKey()
    testForward(mod, {M, pi}, precision_forward)
    testBackward(mod, {M, pi}, precision_backward)
    
end

function cumemtest.LinearMemoryWriter()
    local dimSize = 5
    local batchSize = 2
    local encoderSize = 3 
    local X = torch.rand(encoderSize, batchSize, dimSize)
    local mod = nn.LinearMemoryWriter(dimSize)

    testForward(mod, X, precision_forward)
    testBackward(mod, X, precision_backward)

end


function cumemtest.BilinearAttentionMemoryWriter()
    local encoderSize = 4
    local batchSize = 2
    local dimSize = 10
    local X = torch.rand(encoderSize, batchSize, dimSize)
    local mod = nn.BilinearAttentionMemoryWriter(dimSize)
    testForward(mod, X, precision_forward)
    testBackward(mod, X, precision_backward)

end

function cumemtest.MemoryCell()
    local encoderSize = 4
    local batchSize = 2
    local dimSize = 10
    local X = torch.rand(encoderSize, batchSize, dimSize)

    local cell = nn.MemoryCell():add(nn.LinearAssociativeMemoryWriterP(dimSize))
    local net = nn.Sequential():add(cell):add(nn.SortOnKey(true)):add(
    nn.ParallelTable():add(nn.Identity()):add(nn.Sequential():add(
        nn.Replicate(dimSize, 3, 2)))
    ):add(nn.CMulTable()):add(nn.Sum(1,3,false))


    testForward(net, X, precision_forward)
    testBackward(net, X, precision_backward)

end

function cumemtest.CoupledLSTM()
    local numLayers = 2
    local encoderSize = 5
    local decoderSize = 6
    local batchSize = 2
    local dimSize = 10
    local X = torch.rand(encoderSize, batchSize, dimSize)
    local Y = torch.rand(decoderSize, batchSize, dimSize)

    local mod = nn.CoupledLSTM(dimSize, numLayers)
    testForward(mod, {X, Y}, precision_forward)
    testBackward(mod, {X, Y}, precision_backward)
    
    mod:decouple()
    testForward(mod, {X, Y}, precision_forward)
    testBackward(mod, {X, Y}, precision_backward)

end    

local function setUp()
   cutorch.setDevice(1)
end

for k,v in pairs(cumemtest.__tests) do
   cumemtest.__tests[k] = function()
      setUp()
      v()
   end
end

local function initSeed(seed)
   seed = seed or math.floor((torch.tic() * 1e5) % 1e9)
   -- ensure that you can reproduce a failing test
   print('seed: ', seed)
   math.randomseed(seed)
   torch.manualSeed(seed)
   cutorch.manualSeedAll(seed)
end

function nn.MemoryTestCuda(tests, seed)
   local oldtype = torch.getdefaulttensortype()
   torch.setdefaulttensortype('torch.FloatTensor')
   initSeed(seed)
   mytester:add(cumemtest)
   mytester:run(tests)
   torch.setdefaulttensortype(oldtype)
end
