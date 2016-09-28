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
    local params, gradParams = protoModule:getParameters()
    local output = protoModule:forward(input)
    local outputGrad = makeGrad(output)
    local inputGrad = protoModule:backward(input, outputGrad)   
 
    local cudaMod = protoModule:clone():cuda()
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

    local cell = nn.MemoryCell():add(nn.LinearMemoryWriter(dimSize)):add(
        nn.BilinearAttentionMemoryWriter(dimSize))
    local net = nn.Sequential():add(cell):add(nn.SortOnKey(true)):add(
    nn.ParallelTable():add(nn.Identity()):add(nn.Sequential():add(
        nn.Replicate(dimSize, 3, 2)))
    ):add(nn.CMulTable()):add(nn.Sum(1,3,false))


    testForward(net, X, precision_forward)
    testBackward(net, X, precision_backward)


    local batchSize = 2
    local maxSteps = 3
    local X = torch.rand(maxSteps, batchSize, dimSize):cuda()
    local Y = torch.rand(2, batchSize, dimSize):cuda()

    X[2][1]:fill(0)
    X[3][2]:fill(0)
    Y[1]:copy(X[1])
    Y[2][1] = X[3][1]
    Y[2][2] = X[2][2]


    local cell = nn.MemoryCell():add(nn.LinearMemoryWriter(dimSize)):add(
        nn.LinearMemoryWriter(dimSize))
    cell:maskZero()

    local net = nn.Sequential():add(cell):add(nn.SortOnKey(true)):add(
    nn.ParallelTable():add(nn.Identity()):add(nn.Sequential():add(
        nn.Replicate(dimSize, 3, 2)))
    ):add(nn.CMulTable()):add(nn.Sum(1,3,false)):cuda()

    local G = torch.rand(batchSize, dimSize):cuda()

    local params, gradParams = net:getParameters()

    net:zeroGradParameters()
    local outputX = net:forward(X)
    local gradX = net:backward(X, G):clone()
    local gradXparams = gradParams:clone()

    net:zeroGradParameters()
    local outputY = net:forward(Y)
    local gradY = net:backward(Y, G):clone()
    local gradYparams = gradParams:clone()

    mytester:asserteq(outputX, outputY)
    mytester:assertTensorEq(gradX[1], gradY[1])
    mytester:assertTensorEq(gradX[3][1], gradY[2][1])
    mytester:assertTensorEq(gradX[2][1], torch.zeros(dimSize):cuda())
    mytester:assertTensorEq(gradX[3][2], torch.zeros(dimSize):cuda())
    mytester:assertTensorEq(gradX[2][2], gradY[2][2])
    mytester:assertTensorEq(gradXparams, gradYparams, 1e-7)



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
