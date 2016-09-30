local mytester = torch.Tester()
local precision = 1e-5
local expprecision = 1e-4
local jac

local memtest = torch.TestSuite()

function memtest.LinearAssociativeMemoryReader()

    local dimSize = 5
    local batchSize = 2
    local encoderSizeMask = 3
    local decoderSizeMask = 3
    local encoderSize = 2
    local decoderSize = 2

    local Dmask = torch.rand(encoderSizeMask + decoderSizeMask, 
        batchSize, dimSize)
    Dmask[{{2},{1},{}}] = 0
    Dmask[{{3},{2},{}}] = 0
    Dmask[{{4},{1},{}}] = 0
    Dmask[{{5},{2},{}}] = 0
    
    local D = torch.Tensor(encoderSize + decoderSize, batchSize, dimSize)
    D[{{1},{},{}}]:copy(Dmask[{{1},{},{}}])
    D[{{2},{2},{}}]:copy(Dmask[{{2},{2},{}}])
    D[{{2},{1},{}}]:copy(Dmask[{{3},{1},{}}])
    D[{{3},{1},{}}]:copy(Dmask[{{5},{1},{}}])
    D[{{4},{1},{}}]:copy(Dmask[{{6},{1},{}}])
    D[{{3},{2},{}}]:copy(Dmask[{{4},{2},{}}])
    D[{{4},{2},{}}]:copy(Dmask[{{6},{2},{}}])

    local mod = nn.LinearAssociativeMemoryReader(dimSize)
    local net = nn.Sequential():add(
        nn.ConcatTable():add(    
            nn.Narrow(1,1,encoderSize)):add(
            nn.Narrow(1,encoderSize+1,decoderSize))):add(
        mod)

    local params, gradParams = net:getParameters()

    local modMask = nn.LinearAssociativeMemoryReader(dimSize):maskZero()
    local netMask = nn.Sequential():add(
        nn.ConcatTable():add(    
            nn.Narrow(1,1,encoderSizeMask)):add(
            nn.Narrow(1,encoderSizeMask+1,decoderSizeMask))):add(
        modMask)

    local paramsMask, gradParamsMask = netMask:getParameters()
    paramsMask:copy(params)
    net:zeroGradParameters()
    netMask:zeroGradParameters()

    local outputMask = netMask:forward(Dmask)
    local output = net:forward(D)
    local gradOutputMask = torch.Tensor(
        decoderSizeMask, batchSize, dimSize):zero()
    local gradOutput = torch.rand(decoderSize, batchSize, dimSize)
    gradOutputMask[{{2},{1},{}}] = gradOutput[{{1},{1},{}}]
    gradOutputMask[{{3},{1},{}}] = gradOutput[{{2},{1},{}}]
    gradOutputMask[{{1},{2},{}}] = gradOutput[{{1},{2},{}}]
    gradOutputMask[{{3},{2},{}}] = gradOutput[{{2},{2},{}}]
    local gradInputMask = netMask:backward(Dmask, gradOutputMask)
    local gradInput = net:backward(D, gradOutput)

    mytester:assertTensorEq(
        output[{{1},{1},{}}], outputMask[{{2},{1},{}}], precision)
    mytester:assertTensorEq(
        output[{{2},{1},{}}], outputMask[{{3},{1},{}}], precision)
    mytester:assertTensorEq(
        output[{{1},{2},{}}], outputMask[{{1},{2},{}}], precision)
    mytester:assertTensorEq(
        output[{{2},{2},{}}], outputMask[{{3},{2},{}}], precision)

    mytester:assertTensorEq(
        gradInput[{{1},{1},{}}], gradInputMask[{{1},{1},{}}], precision)
    mytester:assertTensorEq(
        torch.Tensor(1,1,dimSize):zero(), 
        gradInputMask[{{2},{1},{}}], precision)
    mytester:assertTensorEq(
        gradInput[{{2},{1},{}}], gradInputMask[{{3},{1},{}}], precision)

    mytester:assertTensorEq(
        gradInput[{{1},{2},{}}], gradInputMask[{{1},{2},{}}], precision)
    mytester:assertTensorEq(
        torch.Tensor(1,1,dimSize):zero(), 
        gradInputMask[{{3},{2},{}}], precision)
    mytester:assertTensorEq(
        gradInput[{{2},{2},{}}], gradInputMask[{{2},{2},{}}], precision)
 

    mytester:assertTensorEq(
        torch.Tensor(1,1,dimSize):zero(), 
        gradInputMask[{{4},{1},{}}], precision)
    mytester:assertTensorEq(
        gradInput[{{3},{1},{}}], gradInputMask[{{5},{1},{}}], precision)
    mytester:assertTensorEq(
        gradInput[{{4},{1},{}}], gradInputMask[{{6},{1},{}}], precision)

    mytester:assertTensorEq(
        torch.Tensor(1,1,dimSize):zero(), 
        gradInputMask[{{5},{2},{}}], precision)
    mytester:assertTensorEq(
        gradInput[{{3},{2},{}}], gradInputMask[{{4},{2},{}}], precision)
    mytester:assertTensorEq(
        gradInput[{{4},{2},{}}], gradInputMask[{{6},{2},{}}], precision)

    mytester:assertTensorEq(
        gradParamsMask, gradParams, precision)

end

mytester:add(memtest)

jac = nn.Jacobian
function nn.MemoryMaskTest(tests, seed)
   -- Limit number of threads since everything is small
   local nThreads = torch.getnumthreads()
   torch.setnumthreads(1)
   -- randomize stuff
   local seed = seed or (1e5 * torch.tic())
   print('Seed: ', seed)
   math.randomseed(seed)
   torch.manualSeed(seed)
   mytester:run(tests)
   torch.setnumthreads(nThreads)
   return mytester
end

