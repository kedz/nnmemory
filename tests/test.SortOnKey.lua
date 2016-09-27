require 'nn'
require 'memory'

local tester = torch.Tester()

local tolerance = .000000001
local DIM_SIZE = 5
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


local pqtests = torch.TestSuite()

function pqtests.SortOnKeyTestOutputAscending()

    local maxSteps = 4
    local batchSize = 2
    local dimSize = 3
    local M = torch.Tensor(maxSteps, batchSize, dimSize)
    local pi = torch.Tensor(maxSteps, batchSize)
    M[1][1]:fill(3)
    M[2][1]:fill(4)
    M[3][1]:fill(2)
    M[4][1]:fill(1)
    M[1][2]:fill(2)
    M[2][2]:fill(4)
    M[3][2]:fill(1)
    M[4][2]:fill(3)
    pi[1][1] = 3
    pi[2][1] = 4
    pi[3][1] = 2
    pi[4][1] = 1
    pi[1][2] = 2
    pi[2][2] = 4
    pi[3][2] = 1
    pi[4][2] = 3

    local Msorted = torch.Tensor(maxSteps, batchSize, dimSize)
    local piSorted = torch.Tensor(maxSteps, batchSize)
    for i=1,maxSteps do 
        Msorted[i]:fill(i)
        piSorted[i]:fill(i)
    end

    local net = nn.SortOnKey()
    local output = net:forward({M, pi})
    tester:assertTensorEq(output[1], Msorted, 0, "Sorting memory failed.")
    tester:assertTensorEq(output[2], piSorted, 0, "Sorting key failed.")

end

function pqtests.SortOnKeyTestOutputDescending()

    local maxSteps = 4
    local batchSize = 2
    local dimSize = 3
    local M = torch.Tensor(maxSteps, batchSize, dimSize)
    local pi = torch.Tensor(maxSteps, batchSize)
    M[1][1]:fill(3)
    M[2][1]:fill(4)
    M[3][1]:fill(2)
    M[4][1]:fill(1)
    M[1][2]:fill(2)
    M[2][2]:fill(4)
    M[3][2]:fill(1)
    M[4][2]:fill(3)
    pi[1][1] = 3
    pi[2][1] = 4
    pi[3][1] = 2
    pi[4][1] = 1
    pi[1][2] = 2
    pi[2][2] = 4
    pi[3][2] = 1
    pi[4][2] = 3

    local Msorted = torch.Tensor(maxSteps, batchSize, dimSize)
    local piSorted = torch.Tensor(maxSteps, batchSize)
    for i=1,maxSteps do 
        Msorted[i]:fill(maxSteps - i + 1)
        piSorted[i]:fill(maxSteps - i + 1)
    end

    local net = nn.SortOnKey(true)
    local output = net:forward({M, pi})
    tester:assertTensorEq(output[1], Msorted, 0, "Sorting memory failed.")
    tester:assertTensorEq(output[2], piSorted, 0, "Sorting key failed.")

end


function pqtests.SortOnKeyTestBackpropMemory()

    local maxSteps = ENCODER_SIZE
    local batchSize = BATCH_SIZE
    local dimSize = DIM_SIZE
    local M = torch.Tensor(maxSteps, batchSize, dimSize)
    local pi = torch.rand(maxSteps, batchSize)
    local gradPi = torch.Tensor(maxSteps, batchSize):zero()

    local mod = nn.SortOnKey()

    local net = {} 
    function net:forward(M)
        self.output_table = mod:forward({M, pi})
        self.output = self.output_table[1]
        return self.output
    end
    function net:zeroGradParameters()
        mod:zeroGradParameters()
    end
    function net:backward(M, gradOutput)
        return mod:backward({M,pi}, {gradOutput, gradPi})
    end
    function net:updateGradInput(M, gradOutput)
        self.gradInput = mod:updateGradInput({M, pi}, {gradOutput, gradPi})
        return self.gradInput[1]
    end
    function net:accGradParameters(M, gradOutput)
        mod:accGradParameters({M, pi}, {gradOutput, gradPi})
    end

    local err = nn.Jacobian.testJacobian(net, M)
    tester:assertalmosteq(
        err, 0, tolerance, 
        "SortOnKey memory gradient not computed correctly.")
 

end 

function pqtests.SortOnKeyTestBackpropKey()

    local maxSteps = ENCODER_SIZE
    local batchSize = BATCH_SIZE
    local dimSize = DIM_SIZE
    local M = torch.Tensor(maxSteps, batchSize, dimSize)
    local pi = torch.rand(maxSteps, batchSize)
    local gradM = torch.Tensor(maxSteps, batchSize, dimSize):zero()

    local mod = nn.SortOnKey()

    local net = {} 
    function net:forward(pi)
        self.output_table = mod:forward({M, pi})
        self.output = self.output_table[2]
        return self.output
    end
    function net:zeroGradParameters()
        mod:zeroGradParameters()
    end
    function net:backward(pi, gradOutput)
        return mod:backward({M,pi}, {gradM, gradOutput})
    end
    function net:updateGradInput(pi, gradOutput)
        self.gradInput = mod:updateGradInput({M, pi}, {gradM, gradOutput})
        return self.gradInput[2]
    end
    function net:accGradParameters(pi, gradOutput)
        mod:accGradParameters({M, pi}, {gradM, gradOutput})
    end

    local err = nn.Jacobian.testJacobian(net, pi)
    tester:assertalmosteq(
        err, 0, tolerance, 
        "SortOnKey key gradient not computed correctly.")
end 

tester:add(pqtests)
tester:run()
