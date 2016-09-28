
local mytester = torch.Tester()
local precision = 1e-5
local expprecision = 1e-4
local jac

local memtest = torch.TestSuite()


function memtest.SortOnKey()
    local encoderSize = 4
    local batchSize = 2
    local dimSize = 10
    local M = torch.rand(encoderSize, batchSize, dimSize)
    local pi = torch.rand(encoderSize, batchSize)

    local mod = nn.Sequential():add(nn.SortOnKey()):add(
        nn.ParallelTable():add(nn.Identity()):add(nn.Sequential():add(
            nn.Replicate(dimSize, 3, 2)))
        ):add(nn.CMulTable()):add(nn.Sum(1,3,false))


    local netM = {} 
    function netM:forward(M)
        self.output = mod:forward({M, pi})
        return self.output
    end
    function netM:zeroGradParameters()
        mod:zeroGradParameters()
    end
    function netM:backward(M, gradOutput)
        return mod:backward({M,pi}, gradOutput)
    end
    function netM:updateGradInput(M, gradOutput)
        self.gradInput = mod:updateGradInput({M, pi}, gradOutput)
        return self.gradInput[1]
    end
    function netM:accGradParameters(M, gradOutput)
        mod:accGradParameters({M, pi}, gradOutput)
    end

    local err = nn.Jacobian.testJacobian(netM, M)
    mytester:assertlt(
        err, precision, 
        "SortOnKey memory gradient not computed correctly.")
 
    local netPi = {} 
    function netPi:forward(pi)
        self.output = mod:forward({M, pi})
        return self.output
    end
    function netPi:zeroGradParameters()
        mod:zeroGradParameters()
    end
    function netPi:backward(pi, gradOutput)
        return mod:backward({M,pi}, gradOutput)
    end
    function netPi:updateGradInput(pi, gradOutput)
        self.gradInput = mod:updateGradInput({M, pi}, gradOutput)
        return self.gradInput[2]
    end
    function netPi:accGradParameters(pi, gradOutput)
        mod:accGradParameters({M, pi}, gradOutput)
    end

    local err = nn.Jacobian.testJacobian(netPi, pi)
    mytester:assertlt(
        err, precision, 
        "SortOnKey key gradient not computed correctly.")

end

function memtest.LinearMemoryWriter()
    local encoderSize = 4
    local batchSize = 2
    local dimSize = 10
    local X = torch.rand(encoderSize, batchSize, dimSize)
    local mod = nn.LinearMemoryWriter(dimSize)

    local err = nn.Jacobian.testJacobian(mod, X)
    mytester:assertlt(
        err, precision,
        "LinearMemoryWriter input gradient not computed correctly.")
    local params, gradParams = mod:getParameters()

    local errParam = nn.Jacobian.testJacobianParameters(
        mod, X, params, gradParams)
    mytester:assertlt(
        errParam, precision,
        "LinearMemoryWriter parameter gradient not computed correctly.")

end

function memtest.BilinearAttentionMemoryWriter()
    local encoderSize = 4
    local batchSize = 2
    local dimSize = 50
    local X = torch.rand(encoderSize, batchSize, dimSize)
    local mod = nn.BilinearAttentionMemoryWriter(dimSize)

    local err = jac.testJacobian(mod, X)
    mytester:assertlt(err, precision,
        "BilinearAttentionMemoryWriter input gradient is incorrect. " .. err)
end

function memtest.MemoryCell()
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

    local err = nn.Jacobian.testJacobian(net, X)
    mytester:assertlt(
            err, precision, 
        "MemoryCell input gradient not computed correctly.")

    local params, gradParams = net:getParameters()

    local errParam = nn.Jacobian.testJacobianParameters(
        net, X, params, gradParams)
    mytester:assertlt(
        errParam, precision,
        "MemoryCell parameter gradient not computed correctly.")

    local batchSize = 2
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

    local net = nn.Sequential():add(cell):add(nn.SortOnKey(true)):add(
    nn.ParallelTable():add(nn.Identity()):add(nn.Sequential():add(
        nn.Replicate(dimSize, 3, 2)))
    ):add(nn.CMulTable()):add(nn.Sum(1,3,false))

    local G = torch.rand(batchSize, dimSize)

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
    mytester:assertTensorEq(gradX[2][1], torch.zeros(dimSize))
    mytester:assertTensorEq(gradX[3][2], torch.zeros(dimSize))
    mytester:assertTensorEq(gradX[2][2], gradY[2][2])
    mytester:assertTensorEq(gradXparams, gradYparams)

end

function memtest.CoupledLSTM()
    local numLayers = 2
    local encoderSize = 4
    local batchSize = 2
    local dimSize = 5
    local input = torch.rand(2, encoderSize, batchSize, dimSize)
    local mod = nn.CoupledLSTM(dimSize, numLayers)
    local net = nn.Sequential():add(nn.SplitTable(1, 4)):add(mod):add(
        nn.JoinTable(1,3))

    local err = nn.Jacobian.testJacobian(net, input)
    mytester:assertlt(err, precision, 
        "CoupledLSTM input gradient is incorrect.")
    local params, gradParams = mod:getParameters()

    net:zeroGradParameters()
    local errParam = nn.Jacobian.testJacobianParameters(
        net, input, params, gradParams)
    mytester:assertlt(
        errParam, precision,
        "CoupledLSTM parameter gradient not computed correctly.")

    mod:decouple()
    net:zeroGradParameters()
    local err = nn.Jacobian.testJacobian(net, input)
    mytester:assertlt(err, precision, 
        "CoupledLSTM input gradient is incorrect.")
    local params, gradParams = mod:getParameters()

    net:zeroGradParameters()
    local errParam = nn.Jacobian.testJacobianParameters(
        net, input, params, gradParams)
    mytester:assertlt(
        errParam, precision,
        "CoupledLSTM parameter gradient not computed correctly.")
       
end


 

mytester:add(memtest)

jac = nn.Jacobian
function nn.MemoryTest(tests, seed)
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

