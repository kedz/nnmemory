require 'nn'
require 'memory'

local tester = torch.Tester()

local tolerance = .000000001
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




dimSize = 2
batchSize = 4
encoderSize = 3
decoderSize = 5

local X = torch.rand(encoderSize, batchSize, dimSize)
local Y = torch.rand(decoderSize, batchSize, dimSize)
local priority = torch.Tensor{{0.3421, 0.3645, 0.4010, 0.4519},
                              {0.3341, 0.3338, 0.3310, 0.3230},
                              {0.3238, 0.3017, 0.2680, 0.2251}}


local pqtests = torch.TestSuite()


function pqtests.PriorityQueueSimpleEncoderTestInputWrtM()
    local dimSize = DIM_SIZE
    local batchSize = BATCH_SIZE
    local encoderSize = ENCODER_SIZE
    local decoderSize = DECODER_SIZE
    local input = randomData(encoderSize, decoderSize, batchSize, dimSize)
    local X = input[1] 
    local pi = input[2] 
    local grad_pi = torch.Tensor():resizeAs(pi):zero()

    local qenc = nn.PriorityQueueSimpleEncoder(dimSize)

    local net = {} 
    function net:forward(X)
        self.output_table = qenc:forward(X)
        self.output = self.output_table[1]
        return self.output
    end
    function net:zeroGradParameters()
        qenc:zeroGradParameters()
    end
    function net:backward(X, gradOutput)
        return qenc:backward(X, {gradOutput, grad_pi})
    end
    function net:updateGradInput(X, gradOutput)
        return qenc:updateGradInput(X, {gradOutput, grad_pi})
    end

    function net:accGradParameters(X, gradOutput)
        qenc:accGradParameters(X, {gradOutput, grad_pi})
    end

    local err = nn.Jacobian.testJacobian(net, X)
    tester:assertalmosteq(
        err, 0, tolerance, 
        "PriorityQueueDecoder memory gradient not computed correctly.")
 
    local err_W = nn.Jacobian.testJacobianParameters(
        net, X, qenc.weight, qenc.gradWeight)
    tester:assertalmosteq(
        err_W, 0, tolerance, 
        "PriorityQueueSimpleEncoder weight gradient not computed correctly.")
    
    local err_bias = nn.Jacobian.testJacobianParameters(
        net, X, qenc.bias, qenc.gradBias)
    tester:assertalmosteq(
        err_bias, 0, tolerance, 
        "PriorityQueueSimpleEncoder bias gradient not computed correctly.")

end

function pqtests.PriorityQueueSimpleEncoderTestInputWrtPi()
    local dimSize = DIM_SIZE
    local batchSize = BATCH_SIZE
    local encoderSize = ENCODER_SIZE
    local decoderSize = DECODER_SIZE
    local input = randomData(encoderSize, decoderSize, batchSize, dimSize)
    local X = input[1] 
    local grad_M = torch.Tensor():resizeAs(X):zero()


    local qenc = nn.PriorityQueueSimpleEncoder(dimSize)

    local net = {} 
    function net:forward(X)
        self.output_table = qenc:forward(X)
        self.output = self.output_table[2]
        return self.output
    end
    function net:zeroGradParameters()
        qenc:zeroGradParameters()
    end
    function net:backward(X, gradOutput)
        return qenc:backward(X, {grad_M, self.gradOutput})
    end
    function net:updateGradInput(X, gradOutput)
        return qenc:updateGradInput(X, {grad_M, gradOutput})
    end
    function net:accGradParameters(X, gradOutput)
        qenc:accGradParameters(X, {grad_M, gradOutput})
    end

    local err = nn.Jacobian.testJacobian(net, X)
    tester:assertalmosteq(
        err, 0, tolerance, 
        "PriorityQueueDecoder memory gradient not computed correctly.")
     
    local err_W = nn.Jacobian.testJacobianParameters(
        net, X, qenc.weight, qenc.gradWeight)
    tester:assertalmosteq(
        err_W, 0, tolerance, 
        "PriorityQueueSimpleEncoder weight gradient not computed correctly.")
    
    local err_bias = nn.Jacobian.testJacobianParameters(
        net, X, qenc.bias, qenc.gradBias)
    tester:assertalmosteq(
        err_bias, 0, tolerance, 
        "PriorityQueueSimpleEncoder bias gradient not computed correctly.")

end

function pqtests.PriorityQueueSimpleEncoderTestZeroGradParameters()
    local dimSize = DIM_SIZE
    local batchSize = BATCH_SIZE
    local encoderSize = ENCODER_SIZE
    local decoderSize = DECODER_SIZE
    local input = randomData(encoderSize, decoderSize, batchSize, dimSize)
    local X, pi, Y = unpack(input)

    local qenc = nn.PriorityQueueSimpleEncoder(dimSize)
    local zeroWeight = torch.Tensor():resizeAs(qenc.weight):zero()
    local zeroBias = torch.Tensor():resizeAs(qenc.bias):zero()
 
    local randomMemGrad = torch.randn(encoderSize, batchSize, dimSize)
    local randomPiGrad = torch.randn(encoderSize, batchSize)
    
    local output = qenc:forward(X)
    qenc:backward(X, {randomMemGrad, randomPiGrad})
    
    tester:assertTensorNe(qenc.gradWeight, zeroWeight)
    tester:assertTensorNe(qenc.gradBias, zeroBias)

    qenc:zeroGradParameters()
    tester:assertTensorEq(qenc.gradWeight, zeroWeight)
    tester:assertTensorEq(qenc.gradBias, zeroBias)

end

function pqtests.PriorityQueueSimpleEncoderTestParameters()
    local dimSize = 3

    local zero = torch.Tensor(dimSize + 1):zero()
    local qenc = nn.PriorityQueueSimpleEncoder(dimSize)

    local X = torch.randn(5,2,dimSize)
    local randomMemGrad = torch.randn(5, 2, dimSize)
    local randomPiGrad = torch.randn(5, 2)
    
    local output = qenc:forward(X)
    qenc:backward(X, {randomMemGrad, randomPiGrad})

    local params, gradParams = qenc:parameters()
    tester:asserteq(params[1], qenc.weight)
    tester:asserteq(params[2], qenc.bias)
    tester:asserteq(gradParams[1], qenc.gradWeight)
    tester:asserteq(gradParams[2], qenc.gradBias)

    local flatParams, flatGradParams = qenc:getParameters()

    local output = qenc:forward(X)
    qenc:backward(X, {randomMemGrad, randomPiGrad})

    qenc:zeroGradParameters()
    tester:assertTensorEq(flatGradParams, zero)

end

function pqtests.PriorityQueueSimpleEncoderTestMaskedInput()

    local dimSize = 4
    local X = torch.randn(5,2,dimSize)
    X[1][1]:fill(0)
    X[2][1]:fill(0)

    local qenc = nn.PriorityQueueSimpleEncoder(dimSize):maskZero()
    local output = qenc:forward(X)
    
    -- Check that outputs are masked corresponding to masked inputs
    tester:assertTensorEq(output[1][4][1], X[1][1])
    tester:assertTensorEq(output[1][5][1], X[1][1])
    tester:asserteq(output[2][4][1], 0)
    tester:asserteq(output[2][5][1], 0)

    local Xlong = torch.randn(5,2,dimSize)
    Xlong[1]:fill(0)
    Xlong[2]:fill(0)
    local Xshort = torch.Tensor(3,2,dimSize)
    Xshort:copy(Xlong[{{3,5},{},{}}])
    
    local randomMemGrad = torch.randn(5, 2, dimSize)
    local randomPiGrad = torch.randn(5, 2)
    local Ylong = qenc:forward(Xlong)
    local Glong = {torch.ne(Ylong[1], 0):double(), 
                   torch.ne(Ylong[2], 0):double()}
    torch.cmul(Glong[1], Glong[1], randomMemGrad)
    torch.cmul(Glong[2], Glong[2], randomPiGrad)
    local gradInputLong = qenc:updateGradInput(Xlong, Glong):clone()
    local gradWeightLong = 
        torch.Tensor():resizeAs(qenc.gradWeight):copy(qenc.gradWeight)
    local gradBiasLong = 
        torch.Tensor():resizeAs(qenc.gradBias):copy(qenc.gradBias)

    qenc:zeroGradParameters()

    local Yshort = qenc:forward(Xshort)
    local Gshort = {torch.ne(Yshort[1], 0):double(), 
                    torch.ne(Yshort[2], 0):double()}

    torch.cmul(Gshort[1], Gshort[1], randomMemGrad[{{1,3},{},{}}])
    torch.cmul(Gshort[2], Gshort[2], randomPiGrad[{{1,3},{}}])

    local gradInputShort = qenc:updateGradInput(Xshort, Gshort):clone()
    local gradWeightShort = 
        torch.Tensor():resizeAs(qenc.gradWeight):copy(qenc.gradWeight)
    local gradBiasShort = 
        torch.Tensor():resizeAs(qenc.gradBias):copy(qenc.gradBias)

    -- Check that masked gradients are calculated correctly.
    tester:assertTensorEq(gradInputLong[{{3,5},{},{}}], gradInputShort)
    tester:assertTensorEq(gradWeightLong, gradWeightShort)
    tester:assertTensorEq(gradBiasLong, gradBiasShort)

end

tester:add(pqtests)
tester:run()
