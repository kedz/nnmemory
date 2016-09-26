require 'nn'
require 'memory'

local tester = torch.Tester()

local tolerance = .000000001
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


function pqtests.PriorityQueueSimpleDecoderTestParameters()
    local dimSize = DIM_SIZE
    local batchSize = BATCH_SIZE
    local encoderSize = ENCODER_SIZE
    local decoderSize = DECODER_SIZE
    local input = randomData(encoderSize, decoderSize, batchSize, dimSize)
    local X, pi, Y = unpack(input) 

    local zero = torch.Tensor(4*dimSize + 2):zero()
    local qdec = nn.PriorityQueueSimpleDecoder(dimSize)

    local params, gradParams = qdec:parameters()
    tester:asserteq(#params, 6)
    tester:asserteq(#gradParams, 6)



    local output = qdec:forward(input)
    qdec:backward(input, torch.randn(decoderSize, batchSize, dimSize))

    tester:asserteq(params[1], qdec.weight_read_in)
    tester:asserteq(params[2], qdec.weight_read_h)
    tester:asserteq(params[3], qdec.weight_read_b)
    tester:asserteq(params[4], qdec.weight_forget_in)
    tester:asserteq(params[5], qdec.weight_forget_h)
    tester:asserteq(params[6], qdec.weight_forget_b)

    local flatParams, flatGradParams = qdec:getParameters()
    local output = qdec:forward(input)
    qdec:backward(input, torch.randn(decoderSize, batchSize, dimSize))

    qdec:zeroGradParameters()
    tester:assertTensorEq(flatGradParams, zero)

end

function pqtests.PriorityQueueSimpleDecoderTestZeroGradParameters()
    local dimSize = DIM_SIZE
    local batchSize = BATCH_SIZE
    local encoderSize = ENCODER_SIZE
    local decoderSize = DECODER_SIZE
    local input = randomData(encoderSize, decoderSize, batchSize, dimSize)
    local X, pi, Y = unpack(input) 

    local qdec = nn.PriorityQueueSimpleDecoder(dimSize)
    local output = qdec:forward(input)
    local gradOutput = torch.randn(decoderSize, batchSize, dimSize)
    qdec:backward(input, gradOutput)

    local gradWry = qdec.grad_read_in
    local gradWrh = qdec.grad_read_h
    local gradbr = qdec.grad_read_b
    local gradWfy = qdec.grad_forget_in
    local gradWfh = qdec.grad_forget_h
    local gradbf = qdec.grad_forget_b

    local zeroW = torch.Tensor():resizeAs(gradWry):zero()
    local zerob = torch.Tensor():resizeAs(gradbr):zero()
    tester:assertTensorNe(gradWry, zeroW)
    tester:assertTensorNe(gradWrh, zeroW)
    tester:assertTensorNe(gradbr, zerob)
    tester:assertTensorNe(gradWfy, zeroW)
    tester:assertTensorNe(gradWfh, zeroW)
    tester:assertTensorNe(gradbf, zerob)

    qdec:zeroGradParameters()

    tester:assertTensorEq(gradWry, zeroW)
    tester:assertTensorEq(gradWrh, zeroW)
    tester:assertTensorEq(gradbr, zerob)
    tester:assertTensorEq(gradWfy, zeroW)
    tester:assertTensorEq(gradWfh, zeroW)
    tester:assertTensorEq(gradbf, zerob)

end

function pqtests.PriorityQueueSimpleDecoderTestInputX()
    local dimSize = DIM_SIZE
    local batchSize = BATCH_SIZE
    local encoderSize = ENCODER_SIZE
    local decoderSize = DECODER_SIZE
    local input = randomData(encoderSize, decoderSize, batchSize, dimSize)
    local M, pi, Y = unpack(input) 

    local qdec = nn.PriorityQueueSimpleDecoder(dimSize)

    local net = {} 
    function net:forward(M)
        self.output = qdec:forward({M, pi, Y})
        return self.output
    end
    function net:zeroGradParameters()
        qdec:zeroGradParameters()
    end
    function net:backward(M, gradOutput)
        return qdec:backward({M, pi,Y}, gradOutput)[1]
    end
    function net:updateGradInput(M, gradOutput)
        return qdec:updateGradInput({M, pi, Y}, gradOutput)[1]
    end

    function net:accGradParameters(M, gradOutput)
        qdec:accGradParameters({M, pi, Y}, gradOutput)
    end

    local err = nn.Jacobian.testJacobian(net, M)
    tester:assertalmosteq(
        err, 0, tolerance, 
        "PriorityQueueDecoder memory gradient not computed correctly.")
    
    local err_W_ry = nn.Jacobian.testJacobianParameters(
        net, M, qdec.weight_read_in, qdec.grad_read_in)
    tester:assertalmosteq(
        err_W_ry, 0, tolerance, 
        "PriorityQueueDecoder W_ry gradient not computed correctly.")
    
    local err_W_rh = nn.Jacobian.testJacobianParameters(
        net, M, qdec.weight_read_h, qdec.grad_read_h)
    tester:assertalmosteq(
        err_W_rh, 0, tolerance, 
        "PriorityQueueDecoder W_rh gradient not computed correctly.")
 
    local err_b_r = nn.Jacobian.testJacobianParameters(
        net, M, qdec.weight_read_b, qdec.grad_read_b)
    tester:assertalmosteq(
        err_b_r, 0, tolerance, 
        "PriorityQueueDecoder b_r gradient not computed correctly.")

    local err_W_fy = nn.Jacobian.testJacobianParameters(
        net, M, qdec.weight_forget_in, qdec.grad_forget_in)
    tester:assertalmosteq(
        err_W_fy, 0, tolerance, 
        "PriorityQueueDecoder W_fy gradient not computed correctly.")
    
    local err_W_fh = nn.Jacobian.testJacobianParameters(
        net, M, qdec.weight_forget_h, qdec.grad_forget_h)
    tester:assertalmosteq(
        err_W_fh, 0, tolerance, 
        "PriorityQueueDecoder W_fh gradient not computed correctly.")
 
    local err_b_f = nn.Jacobian.testJacobianParameters(
        net, M, qdec.weight_forget_b, qdec.grad_forget_b)
    tester:assertalmosteq(
        err_b_f, 0, tolerance, 
        "PriorityQueueDecoder b_f gradient not computed correctly.")

end

function pqtests.priorityQueueSimpleDecoderTestDecoderMaskZero()
    local dimSize = 2
    local decoderSize = 5
    local batchSize = 4
    local encoderSize = 3

    local X = torch.rand(encoderSize, batchSize, dimSize)
    local Y = torch.rand(decoderSize, batchSize, dimSize)
    local priority = torch.Tensor{{0.3421, 0.3645, 0.4010, 0.4519},
                                  {0.3341, 0.3338, 0.3310, 0.3230},
                                  {0.3238, 0.3017, 0.2680, 0.2251}}


    local zero = torch.Tensor():resize(dimSize):zero()
    Y[5]:fill(0)
    Y[4]:fill(0)
    Y[3][1]:fill(0)

    local qdec = nn.PriorityQueueSimpleDecoder(dimSize):maskZero()
    local output = qdec:forward({X, priority, Y})

    tester:assertTensorEq(output[5][1], zero)
    tester:assertTensorEq(output[5][2], zero)
    tester:assertTensorEq(output[5][3], zero)
    tester:assertTensorEq(output[5][4], zero)
    tester:assertTensorEq(output[4][1], zero)
    tester:assertTensorEq(output[4][2], zero)
    tester:assertTensorEq(output[4][3], zero)
    tester:assertTensorEq(output[4][4], zero)
    tester:assertTensorEq(output[3][1], zero)

    local Yshort = torch.rand(decoderSize-2, batchSize, dimSize)
    local Ylong = torch.Tensor():resizeAs(Y):fill(0)
    Ylong[{{1,decoderSize-2},{},{}}]:copy(Yshort)
    local rand = torch.randn(decoderSize, batchSize, dimSize)
    local Glong = torch.Tensor():resizeAs(Ylong)
    torch.cmul(Glong, torch.ne(Ylong, 0):double(), rand)
    local Gshort = torch.Tensor():resizeAs(Yshort) 
    Gshort:copy(Glong[{{1,decoderSize-2},{},{}}])
   
    qdec:forward({X,priority,Ylong})
    local gradLong = qdec:updateGradInput({X, priority, Ylong}, Glong)
    local gradMLong = torch.Tensor():resizeAs(gradLong[1]):copy(gradLong[1])
    local gradPiLong = torch.Tensor():resizeAs(gradLong[2]):copy(gradLong[2])
    local gradYLong = torch.Tensor():resizeAs(gradLong[3]):copy(gradLong[3])
    local gradWryLong = 
        torch.Tensor():resizeAs(qdec.grad_read_in):copy(qdec.grad_read_in)
    local gradWrhLong = 
        torch.Tensor():resizeAs(qdec.grad_read_h):copy(qdec.grad_read_h)
    local gradbrLong = 
        torch.Tensor():resizeAs(qdec.grad_read_b):copy(qdec.grad_read_b)
    local gradWfyLong = 
        torch.Tensor():resizeAs(qdec.grad_forget_in):copy(qdec.grad_forget_in)
    local gradWfhLong = 
        torch.Tensor():resizeAs(qdec.grad_forget_h):copy(qdec.grad_forget_h)
    local gradbfLong = 
        torch.Tensor():resizeAs(qdec.grad_forget_b):copy(qdec.grad_forget_b)

    qdec:zeroGradParameters()

    qdec:forward({X,priority,Yshort})
    local gradShort = qdec:updateGradInput({X, priority, Yshort}, Gshort)
    local gradMShort = torch.Tensor():resizeAs(gradShort[1]):copy(gradShort[1])
    local gradPiShort = 
        torch.Tensor():resizeAs(gradShort[2]):copy(gradShort[2])
    local gradYShort = torch.Tensor():resizeAs(gradShort[3]):copy(gradShort[3])
    local gradWryShort = 
        torch.Tensor():resizeAs(qdec.grad_read_in):copy(qdec.grad_read_in)
    local gradWrhShort = 
        torch.Tensor():resizeAs(qdec.grad_read_h):copy(qdec.grad_read_h)
    local gradbrShort = 
        torch.Tensor():resizeAs(qdec.grad_read_b):copy(qdec.grad_read_b)
    local gradWfyShort = 
        torch.Tensor():resizeAs(qdec.grad_forget_in):copy(qdec.grad_forget_in)
    local gradWfhShort = 
        torch.Tensor():resizeAs(qdec.grad_forget_h):copy(qdec.grad_forget_h)
    local gradbfShort = 
        torch.Tensor():resizeAs(qdec.grad_forget_b):copy(qdec.grad_forget_b)


    tester:assertTensorEq(gradMLong, gradMShort)
    tester:assertTensorEq(gradPiLong, gradPiShort)
    tester:assertTensorEq(gradYLong[{{1,decoderSize-2},{},{}}], gradYShort)
    tester:assertTensorEq(gradWryLong, gradWryShort)
    tester:assertTensorEq(gradWrhLong, gradWrhShort)
    tester:assertTensorEq(gradbrLong, gradbrShort)
    tester:assertTensorEq(gradWfyLong, gradWfyShort)
    tester:assertTensorEq(gradWfhLong, gradWfhShort)
    tester:assertTensorEq(gradbfLong, gradbfShort)
end




function pqtests.priorityQueueSimpleDecoderTestInputPi()
    local dimSize = DIM_SIZE
    local batchSize = BATCH_SIZE
    local encoderSize = ENCODER_SIZE
    local decoderSize = DECODER_SIZE
    local input = randomData(encoderSize, decoderSize, batchSize, dimSize)
    local X, pi, Y = unpack(input) 

    local qdec = nn.PriorityQueueSimpleDecoder(dimSize)

    local net = {} 
    function net:forward(pi)
        self.output = qdec:forward({X, pi, Y})
        return self.output
    end
    function net:zeroGradParameters()
        qdec:zeroGradParameters()
    end
    function net:backward(pi, gradOutput)
        return qdec:backward({X, pi, Y}, gradOutput)[2]
    end
    function net:updateGradInput(pi, gradOutput)
        return qdec:updateGradInput({X, pi, Y}, gradOutput)[2]
    end
    function net:accGradParameters(pi, gradOutput)
        qdec:accGradParameters({X, pi, Y}, gradOutput)
    end

    local err = nn.Jacobian.testJacobian(net, pi, 0, 1)
    tester:assertalmosteq(
        err, 0, tolerance, 
        "PriorityQueueDecoder memory gradient not computed correctly.")
    
    local err_W_ry = nn.Jacobian.testJacobianParameters(
        net, pi, qdec.weight_read_in, qdec.grad_read_in, 0, 1)
    tester:assertalmosteq(
        err_W_ry, 0, tolerance, 
        "PriorityQueueDecoder W_ry gradient not computed correctly.")
    
    local err_W_rh = nn.Jacobian.testJacobianParameters(
        net, pi, qdec.weight_read_h, qdec.grad_read_h, 0, 1)
    tester:assertalmosteq(
        err_W_rh, 0, tolerance, 
        "PriorityQueueDecoder W_rh gradient not computed correctly.")
 
    local err_b_r = nn.Jacobian.testJacobianParameters(
        net, pi, qdec.weight_read_b, qdec.grad_read_b, 0, 1)
    tester:assertalmosteq(
        err_b_r, 0, tolerance, 
        "PriorityQueueDecoder b_r gradient not computed correctly.")

    local err_W_fy = nn.Jacobian.testJacobianParameters(
        net, pi, qdec.weight_forget_in, qdec.grad_forget_in, 0, 1)
    tester:assertalmosteq(
        err_W_fy, 0, tolerance, 
        "PriorityQueueDecoder W_fy gradient not computed correctly.")
    
    local err_W_fh = nn.Jacobian.testJacobianParameters(
        net, pi, qdec.weight_forget_h, qdec.grad_forget_h, 0, 1)
    tester:assertalmosteq(
        err_W_fh, 0, tolerance, 
        "PriorityQueueDecoder W_fh gradient not computed correctly.")
 
    local err_b_f = nn.Jacobian.testJacobianParameters(
        net, pi, qdec.weight_forget_b, qdec.grad_forget_b, 0, 1)
    tester:assertalmosteq(
        err_b_f, 0, tolerance, 
        "PriorityQueueDecoder b_f gradient not computed correctly.")

end

function pqtests.priorityQueueSimpleDecoderTestInputY()
    local dimSize = DIM_SIZE
    local batchSize = BATCH_SIZE
    local encoderSize = ENCODER_SIZE
    local decoderSize = DECODER_SIZE
    local input = randomData(encoderSize, decoderSize, batchSize, dimSize)
    local X, pi, Y = unpack(input) 

    local qdec = nn.PriorityQueueSimpleDecoder(dimSize)
    
    local net = {} 
    function net:forward(Y)
        self.output = qdec:forward({X, pi, Y})
        return self.output
    end
    function net:zeroGradParameters()
        qdec:zeroGradParameters()
    end
    function net:backward(Y, gradOutput)
        return qdec:backward({X, pi, Y}, gradOutput)[3]
    end
    function net:updateGradInput(Y, gradOutput)
        return qdec:updateGradInput({X, pi, Y}, gradOutput)[3]
    end

    function net:accGradParameters(Y, gradOutput)
        qdec:accGradParameters({X, pi, Y}, gradOutput)
    end

    local err = nn.Jacobian.testJacobian(net, Y)
    tester:assertalmosteq(
        err, 0, tolerance, 
        "PriorityQueueDecoder memory gradient not computed correctly.")
    
    local err_W_ry = nn.Jacobian.testJacobianParameters(
        net, Y, qdec.weight_read_in, qdec.grad_read_in)
    tester:assertalmosteq(
        err_W_ry, 0, tolerance, 
        "PriorityQueueDecoder W_ry gradient not computed correctly.")
    
    local err_W_rh = nn.Jacobian.testJacobianParameters(
        net, Y, qdec.weight_read_h, qdec.grad_read_h)
    tester:assertalmosteq(
        err_W_rh, 0, tolerance, 
        "PriorityQueueDecoder W_rh gradient not computed correctly.")
 
    local err_b_r = nn.Jacobian.testJacobianParameters(
        net, Y, qdec.weight_read_b, qdec.grad_read_b)
    tester:assertalmosteq(
        err_b_r, 0, tolerance, 
        "PriorityQueueDecoder b_r gradient not computed correctly.")

    local err_W_fy = nn.Jacobian.testJacobianParameters(
        net, Y, qdec.weight_forget_in, qdec.grad_forget_in)
    tester:assertalmosteq(
        err_W_fy, 0, tolerance, 
        "PriorityQueueDecoder W_fy gradient not computed correctly.")
    
    local err_W_fh = nn.Jacobian.testJacobianParameters(
        net, Y, qdec.weight_forget_h, qdec.grad_forget_h)
    tester:assertalmosteq(
        err_W_fh, 0, tolerance, 
        "PriorityQueueDecoder W_fh gradient not computed correctly.")
 
    local err_b_f = nn.Jacobian.testJacobianParameters(
        net, Y, qdec.weight_forget_b, qdec.grad_forget_b)
    tester:assertalmosteq(
        err_b_f, 0, tolerance, 
        "PriorityQueueDecoder b_f gradient not computed correctly.")

end



tester:add(pqtests)
tester:run()
