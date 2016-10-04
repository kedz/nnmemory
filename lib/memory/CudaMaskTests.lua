local cumemtest = torch.TestSuite()
local precision = 1e-6
local precision_forward = 1e-4
local precision_backward = 1e-2

local mytester = torch.Tester()


function cumemtest.MemoryCell()
    local dimSize = 5
    local batchSize = 2
    local maxSteps = 3

    local Xmask = torch.rand(maxSteps, batchSize, dimSize):cuda()
    local X = torch.rand(2, batchSize, dimSize):cuda()

    Xmask[2][1]:fill(0)
    Xmask[3][2]:fill(0)
    X[1]:copy(Xmask[1])
    X[2][1] = Xmask[3][1]
    X[2][2] = Xmask[2][2]


    local cellMasked = nn.MemoryCell():add(
        nn.LinearAssociativeMemoryWriterP(dimSize))
    cellMasked:maskZero()

    local netMasked = nn.Sequential():add(cellMasked):add(
        nn.SortOnKey(true)):add(
        nn.ParallelTable():add(nn.Identity()):add(nn.Sequential():add(
            nn.Replicate(dimSize, 3, 2)))
        ):add(nn.CMulTable()):add(nn.Sum(1,3,false))
    netMasked:cuda()

    local cell = nn.MemoryCell():add(
        nn.LinearAssociativeMemoryWriterP(dimSize))

    local net = nn.Sequential():add(cell):add(
        nn.SortOnKey(true)):add(
        nn.ParallelTable():add(nn.Identity()):add(nn.Sequential():add(
            nn.Replicate(dimSize, 3, 2)))
        ):add(nn.CMulTable()):add(nn.Sum(1,3,false))
    net:cuda()


    local gradOutput = torch.rand(batchSize, dimSize):cuda()


    local params, gradParams = net:getParameters()
    local paramsMasked, gradParamsMasked = netMasked:getParameters()
    params:copy(paramsMasked)

    net:zeroGradParameters()
    netMasked:zeroGradParameters()

    local outputMasked = netMasked:forward(Xmask)
    local gradInputMasked = netMasked:backward(Xmask, gradOutput)
    local output = net:forward(X)
    local gradInput = net:backward(X, gradOutput)



    mytester:assertTensorEq(output, outputMasked)
    mytester:assertTensorEq(gradParamsMasked, gradParams, precision)

    mytester:assertTensorEq(gradInput[1], gradInputMasked[1])
    mytester:assertTensorEq(gradInput[2][1], gradInputMasked[3][1])
    mytester:assertTensorEq(gradInput[2][2], gradInputMasked[2][2])
    mytester:assertTensorEq(
        gradInputMasked[2][1], torch.zeros(dimSize):cuda())
    mytester:assertTensorEq(
        gradInputMasked[3][2], torch.zeros(dimSize):cuda())

end




function cumemtest.LinearAssociativeMemoryWriter()

    local dimSize = 5
    local batchSize = 2
    local encoderSizeMask = 3
    local encoderSize = 2

    local Xmask = torch.rand(encoderSizeMask, batchSize, dimSize):cuda()
    Xmask[{{2},{1},{}}] = 0
    Xmask[{{3},{2},{}}] = 0
    local X = torch.Tensor(encoderSize, batchSize, dimSize):zero():cuda()
    X[{{1},{},{}}]:copy(Xmask[{{1},{},{}}])
    X[{{2},{1},{}}]:copy(Xmask[{{3},{1},{}}])
    X[{{2},{2},{}}]:copy(Xmask[{{2},{2},{}}])

    local gradOutputMask = torch.Tensor(
        encoderSizeMask, batchSize, dimSize):zero():cuda()
    local gradOutput = torch.rand(encoderSize, batchSize, dimSize):cuda()
    gradOutputMask[{{1},{},{}}] = gradOutput[{{1},{},{}}]
    gradOutputMask[{{3},{1},{}}] = gradOutput[{{2},{1},{}}]
    gradOutputMask[{{2},{2},{}}] = gradOutput[{{2},{2},{}}]

    local netF = nn.LinearAssociativeMemoryWriter(dimSize, "forward"):cuda()
    local netFMasked = nn.LinearAssociativeMemoryWriter(dimSize, "forward"):cuda()
    netFMasked:maskZero()
    
    local paramsF, gradParamsF = netF:getParameters()
    local paramsFMask, gradParamsFMask = netFMasked:getParameters()
    paramsF:copy(paramsFMask)

    local outputFMask = netFMasked:forward(Xmask)
    local gradInputFMask = netFMasked:backward(Xmask, gradOutputMask)

    local outputF = netF:forward(X)
    local gradInputF = netF:backward(X, gradOutput)

    mytester:assertTensorEq(
        outputF[{{1},{},{}}], outputFMask[{{1},{},{}}], precision)
    mytester:assertTensorEq(
        outputFMask[{{2},{1},{}}], torch.CudaTensor(1,1,dimSize):zero(), precision)
    mytester:assertTensorEq(
        outputFMask[{{3},{2},{}}], torch.CudaTensor(1,1,dimSize):zero(), precision)
    mytester:assertTensorEq(
        outputF[{{2},{1},{}}], outputFMask[{{3},{1},{}}], precision)
    mytester:assertTensorEq(
        outputF[{{2},{2},{}}], outputFMask[{{2},{2},{}}], precision)

    mytester:assertTensorEq(
        gradInputF[{{1},{},{}}], 
        gradInputFMask[{{1},{},{}}], 
        precision)
    mytester:assertTensorEq(
        gradInputFMask[{{2},{1},{}}], 
        torch.Tensor(1,1,dimSize):zero():cuda(), 
        precision)
    mytester:assertTensorEq(
        gradInputFMask[{{3},{2},{}}], 
        torch.Tensor(1,1,dimSize):zero():cuda(), precision)
    mytester:assertTensorEq(
        gradInputF[{{2},{1},{}}], 
        gradInputFMask[{{3},{1},{}}], 
        precision)
    mytester:assertTensorEq(
        gradInputF[{{2},{2},{}}], 
        gradInputFMask[{{2},{2},{}}], 
        precision)
    mytester:assertTensorEq(
        gradParamsFMask, gradParamsF, precision)

    local netB = nn.LinearAssociativeMemoryWriter(dimSize, "backward"):cuda()
    local netBMasked = nn.LinearAssociativeMemoryWriter(dimSize, "backward"):cuda()
    netBMasked:maskZero()
    
    local paramsB, gradParamsB = netB:getParameters()
    local paramsBMask, gradParamsBMask = netBMasked:getParameters()
    paramsB:copy(paramsBMask)

    local outputBMask = netBMasked:forward(Xmask)
    local gradInputBMask = netBMasked:backward(Xmask, gradOutputMask)

    local outputB = netB:forward(X)
    local gradInputB = netB:backward(X, gradOutput)

    mytester:assertTensorEq(
        outputB[{{1},{},{}}], outputBMask[{{1},{},{}}], precision)
    mytester:assertTensorEq(
        outputBMask[{{2},{1},{}}], torch.CudaTensor(1,1,dimSize):zero(), precision)
    mytester:assertTensorEq(
        outputBMask[{{3},{2},{}}], torch.CudaTensor(1,1,dimSize):zero(), precision)
    mytester:assertTensorEq(
        outputB[{{2},{1},{}}], outputBMask[{{3},{1},{}}], precision)
    mytester:assertTensorEq(
        outputB[{{2},{2},{}}], outputBMask[{{2},{2},{}}], precision)

    mytester:assertTensorEq(
        gradInputB[{{1},{},{}}], 
        gradInputBMask[{{1},{},{}}], 
        precision)
    mytester:assertTensorEq(
        gradInputBMask[{{2},{1},{}}], 
        torch.CudaTensor(1,1,dimSize):zero(), 
        precision)
    mytester:assertTensorEq(
        gradInputBMask[{{3},{2},{}}], 
        torch.CudaTensor(1,1,dimSize):zero(), precision)
    mytester:assertTensorEq(
        gradInputB[{{2},{1},{}}], 
        gradInputBMask[{{3},{1},{}}], 
        precision)
    mytester:assertTensorEq(
        gradInputB[{{2},{2},{}}], 
        gradInputBMask[{{2},{2},{}}], 
        precision)

    mytester:assertTensorEq(
        gradParamsBMask, gradParamsB, precision)

    local netA = nn.LinearAssociativeMemoryWriter(dimSize, "all"):cuda()
    local netAMasked = nn.LinearAssociativeMemoryWriter(dimSize, "all"):cuda()
    netAMasked:maskZero()
    
    local paramsA, gradParamsA = netA:getParameters()
    local paramsAMask, gradParamsAMask = netAMasked:getParameters()
    paramsA:copy(paramsAMask)

    local outputAMask = netAMasked:forward(Xmask)
    local gradInputAMask = netAMasked:backward(Xmask, gradOutputMask)

    local outputA = netA:forward(X)
    local gradInputA = netA:backward(X, gradOutput)

    mytester:assertTensorEq(
        outputA[{{1},{},{}}], outputAMask[{{1},{},{}}], precision)
    mytester:assertTensorEq(
        outputAMask[{{2},{1},{}}], torch.CudaTensor(1,1,dimSize):zero(), precision)
    mytester:assertTensorEq(
        outputAMask[{{3},{2},{}}], torch.CudaTensor(1,1,dimSize):zero(), precision)
    mytester:assertTensorEq(
        outputA[{{2},{1},{}}], outputAMask[{{3},{1},{}}], precision)
    mytester:assertTensorEq(
        outputA[{{2},{2},{}}], outputAMask[{{2},{2},{}}], precision)

    mytester:assertTensorEq(
        gradInputA[{{1},{},{}}], 
        gradInputAMask[{{1},{},{}}], 
        precision)
    mytester:assertTensorEq(
        gradInputAMask[{{2},{1},{}}], 
        torch.CudaTensor(1,1,dimSize):zero(), 
        precision)
    mytester:assertTensorEq(
        gradInputAMask[{{3},{2},{}}], 
        torch.CudaTensor(1,1,dimSize):zero(), precision)
    mytester:assertTensorEq(
        gradInputA[{{2},{1},{}}], 
        gradInputAMask[{{3},{1},{}}], 
        precision)
    mytester:assertTensorEq(
        gradInputA[{{2},{2},{}}], 
        gradInputAMask[{{2},{2},{}}], 
        precision)
    mytester:assertTensorEq(
        gradParamsAMask, gradParamsA, precision)

end



function cumemtest.LinearAssociativeMemoryReader()

    local dimSize = 5
    local batchSize = 2
    local encoderSizeMask = 3
    local decoderSizeMask = 3
    local encoderSize = 2
    local decoderSize = 2

    local Dmask = torch.rand(encoderSizeMask + decoderSizeMask, 
        batchSize, dimSize):cuda()
    Dmask[{{2},{1},{}}] = 0
    Dmask[{{3},{2},{}}] = 0
    Dmask[{{4},{1},{}}] = 0
    Dmask[{{5},{2},{}}] = 0
    
    local D = torch.CudaTensor(encoderSize + decoderSize, batchSize, dimSize)
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
        mod):cuda()

    local params, gradParams = net:getParameters()

    local modMask = nn.LinearAssociativeMemoryReader(dimSize):maskZero()
    local netMask = nn.Sequential():add(
        nn.ConcatTable():add(    
            nn.Narrow(1,1,encoderSizeMask)):add(
            nn.Narrow(1,encoderSizeMask+1,decoderSizeMask))):add(
        modMask):cuda()

    local paramsMask, gradParamsMask = netMask:getParameters()
    paramsMask:copy(params)
    net:zeroGradParameters()
    netMask:zeroGradParameters()

    local outputMask = netMask:forward(Dmask)
    local output = net:forward(D)
    local gradOutputMask = torch.CudaTensor(
        decoderSizeMask, batchSize, dimSize):zero()
    local gradOutput = torch.rand(decoderSize, batchSize, dimSize):cuda()
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
        torch.CudaTensor(1,1,dimSize):zero(), 
        gradInputMask[{{2},{1},{}}], precision)
    mytester:assertTensorEq(
        gradInput[{{2},{1},{}}], gradInputMask[{{3},{1},{}}], precision)

    mytester:assertTensorEq(
        gradInput[{{1},{2},{}}], gradInputMask[{{1},{2},{}}], precision)
    mytester:assertTensorEq(
        torch.CudaTensor(1,1,dimSize):zero(), 
        gradInputMask[{{3},{2},{}}], precision)
    mytester:assertTensorEq(
        gradInput[{{2},{2},{}}], gradInputMask[{{2},{2},{}}], precision)
 

    mytester:assertTensorEq(
        torch.CudaTensor(1,1,dimSize):zero(), 
        gradInputMask[{{4},{1},{}}], precision)
    mytester:assertTensorEq(
        gradInput[{{3},{1},{}}], gradInputMask[{{5},{1},{}}], precision)
    mytester:assertTensorEq(
        gradInput[{{4},{1},{}}], gradInputMask[{{6},{1},{}}], precision)

    mytester:assertTensorEq(
        torch.CudaTensor(1,1,dimSize):zero(), 
        gradInputMask[{{5},{2},{}}], precision)
    mytester:assertTensorEq(
        gradInput[{{3},{2},{}}], gradInputMask[{{4},{2},{}}], precision)
    mytester:assertTensorEq(
        gradInput[{{4},{2},{}}], gradInputMask[{{6},{2},{}}], precision)

    mytester:assertTensorEq(
        gradParamsMask, gradParams, precision)

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

function nn.MemoryMaskTestCuda(tests, seed)
   local oldtype = torch.getdefaulttensortype()
   torch.setdefaulttensortype('torch.FloatTensor')
   initSeed(seed)
   mytester:add(cumemtest)
   mytester:run(tests)
   torch.setdefaulttensortype(oldtype)
end
