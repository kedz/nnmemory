local PriorityQueueSimpleDecoderV2, Parent = 
    torch.class('nn.PriorityQueueSimpleDecoderV2', 'nn.Module')

function PriorityQueueSimpleDecoderV2:__init(inputSize, readBiasInit, 
        forgetBiasInit)
 
    self.isMasked = false
    self.inputSize = inputSize
    self.readBiasInit = readBiasInit
    self.forgetBiasInit = forgetBiasInit

    self.output = torch.Tensor()
    self.read = torch.Tensor()

    self.controller = nn.Sequential():add(
        nn.ParallelTable():add(
            nn.Linear(inputSize, inputSize, false)):add(
            nn.Linear(inputSize, inputSize, false)):add(
            nn.Linear(inputSize, inputSize, true))):add(
        nn.CAddTable()):add(
        nn.Tanh())
    self.controllerSteps = {}
    self.controllerSteps[1] = self.controller

    self.readLayer = nn.Sequential():add(
        nn.Linear(inputSize, 1)):add(
        nn.Sigmoid())
    self.readLayerSteps = {}
    self.readLayerSteps[1] = self.readLayer

    self.forgetLayer = nn.Sequential():add(
        nn.Linear(inputSize, 1)):add(
        nn.Sigmoid())
    self.forgetLayerSteps = {}
    self.forgetLayerSteps[1] = self.forgetLayer

    self.P = torch.Tensor()
    self.rho = torch.Tensor()
    self.rhocum = torch.Tensor()
    self.phi = torch.Tensor()
    self.phicum = torch.Tensor()

    self.prevQueueSize = 0

    self.lt_mask_base = torch.Tensor(10,10):uniform(-1,1)
    self.di_mask_base = torch.Tensor(10,10):uniform(-1,1)
    self.grad_wrt_pi_vals_base = torch.Tensor()

    self.h_0 = torch.Tensor(1,inputSize):uniform(-1,1) 
    self.grad_h_0 = torch.Tensor(1,inputSize):zero()
    self.read_0 = torch.Tensor()

    self.gradM = torch.Tensor()
    self.gradY = torch.Tensor()
    self.gradP = torch.Tensor()

    self.buffer1 = torch.Tensor()
    self.buffer2 = torch.Tensor()
    self.maskBuffer1 = torch.Tensor()
    self.maskBuffer2 = torch.Tensor()
    self.maskBuffer3 = torch.Tensor()
    self.maskBuffer4 = torch.Tensor()
    self.grad_rho_wrt_r = torch.Tensor()
    self.grad_phi_wrt_f = torch.Tensor()
    self.grad_h_tp1_wrt_read_t = torch.Tensor()
    self.grad_h_tp1_wrt_r = torch.Tensor()
    self.grad_P_tp1_wrt_f = torch.Tensor()

    self:reset()
end

function PriorityQueueSimpleDecoderV2:maskZero()
    self.isMasked = true
    return self
end

function PriorityQueueSimpleDecoderV2:reset()
    self.controller:reset()
    self.readLayer:forget()
    self.forgetLayer:reset()
    if self.readBiasInit ~= nil then
        self.readLayer:get(1).bias:fill(self.readBiasInit)
    end
    if self.forgetBiasInit ~= nil then
        self.forgetLayer:get(1).bias:fill(self.forgetBiasInit)
    end

    self:zeroGradParameters()
end

function PriorityQueueSimpleDecoderV2:zeroGradParameters()

    self.controller:zeroGradParameters()
    self.readLayer:zeroGradParameters()
    self.forgetLayer:zeroGradParameters()
    self.grad_h_0:zero()

end

function PriorityQueueSimpleDecoderV2:parameters()
    if self.params == nil or self.gradParams == nil then

        self.params, self.gradParams = self.controller:parameters()
        
        local readParams, readGradParams = self.readLayer:parameters()
        for i=1,#readParams do
            table.insert(self.params, readParams[i])
            table.insert(self.gradParams, readGradParams[i])
        end
        local forgetParams, forgetGradParams = self.forgetLayer:parameters()
        for i=1,#forgetParams do
            table.insert(self.params, forgetParams[i])
            table.insert(self.gradParams, forgetGradParams[i])
        end
        table.insert(self.params, self.h_0)
        table.insert(self.gradParams, self.grad_h_0)

    end
    return self.params, self.gradParams
end

function PriorityQueueSimpleDecoderV2:getController(step)
    local controller = self.controllerSteps[step]
    if controller == nil then
        controller = self.controller:clone(
            "weight", "bias", "gradWeight", "gradBias")
        self.controllerSteps[step] = controller
    end

    return controller
end

function PriorityQueueSimpleDecoderV2:getReadLayer(step)
    local readLayer = self.readLayerSteps[step]
    if readLayer == nil then
        readLayer = self.readLayer:clone(
            "weight", "bias", "gradWeight", "gradBias")
        self.readLayerSteps[step] = readLayer
    end
    return readLayer
end

function PriorityQueueSimpleDecoderV2:getForgetLayer(step)
    local forgetLayer = self.forgetLayerSteps[step]
    if forgetLayer == nil then
        forgetLayer = self.forgetLayer:clone(
            "weight", "bias", "gradWeight", "gradBias")
        self.forgetLayerSteps[step] = forgetLayer
    end
    return forgetLayer
end





function PriorityQueueSimpleDecoderV2:updateOutput(input)

    M, P_0, Y = unpack(input)
    local queueSize = M:size(1)
    local decoderSize = Y:size(1)
    local dimSize = self.inputSize
    local batchSize = Y:size(2)

    local h_tm1 = self.h_0:expand(batchSize, dimSize)

    local output = self.output:resize(decoderSize, batchSize, dimSize):zero()
    local read = self.read:resize(decoderSize, batchSize, dimSize):zero()
    local P = self.P:resize(decoderSize, batchSize, queueSize):zero()
    P[1] = P_0:t()

    local rhocum = self.rhocum:resize(decoderSize, batchSize, queueSize):zero()
    local rho = self.rho:resize(decoderSize, batchSize, queueSize):zero()
    local phicum = self.phicum:resize(decoderSize, batchSize, queueSize):zero()
    local phi = self.phi:resize(decoderSize, batchSize, queueSize):zero()

    local read_tm1 = torch.sum(
        self.read_0,
        torch.cmul(
            self.buffer2,
            P[1]:view(batchSize, queueSize, 1):expand(
                batchSize, queueSize, dimSize),
            M:transpose(2,1)),
        2):view(batchSize, dimSize)

    local Ymask
    if self.isMasked then
        self.Ymask = torch.eq(Y:select(3, 1), 0):view(
            decoderSize, batchSize, 1):expand(
            decoderSize, batchSize, dimSize)
        Ymask = self.Ymask
    end

    for t=1,decoderSize do
        local controller_t = self:getController(t)
        h_t = controller_t:forward({h_tm1, read_tm1, Y[t]})

        local readLayer_t = self:getReadLayer(t) 
        local forgetLayer_t = self:getForgetLayer(t) 
        local rt = readLayer_t:forward(h_t)
        local ft = forgetLayer_t:forward(h_t)
   
        if self.isMasked then
            h_t:maskedFill(Ymask[t], 0)
            rt:maskedFill(Ymask[t][{{},{1,1}}], 0)
            ft:maskedFill(Ymask[t][{{},{1,1}}], 0)
        end

        local rhocum_t = rhocum[t]
        local rho_t = rho[t]
        local phicum_t = phicum[t]
        local phi_t = phi[t]
        local P_t = P[t]

        rhocum_t:narrow(2, 2, queueSize-1):copy(P_t:narrow(2, 1, queueSize-1))
        torch.cumsum(rhocum_t, rhocum_t, 2)
        rhocum_t:csub(rt:expand(batchSize, queueSize))
        rhocum_t:mul(-1.0)
        rhocum_t:cmax(0.0)
        torch.cmin(rho_t, P_t, rhocum_t)
        torch.cmul(
            self.buffer1,  
            M:transpose(2,1),
            rho_t:view(batchSize, queueSize, 1):expand(
                batchSize, queueSize, dimSize))
               
        torch.sum(
            read[t]:view(batchSize, 1, dimSize), 
            self.buffer1, 2)

        phicum_t:narrow(2, 2, queueSize-1):copy(P_t:narrow(2, 1, queueSize-1))
        torch.cumsum(phicum_t, phicum_t, 2)
        phicum_t:csub(ft:expand(batchSize, queueSize))
        phicum_t:mul(-1.0)
        phicum_t:cmax(0.0)
        torch.cmin(phi_t, P_t, phicum_t)
         
        output[t] = h_t
        read_tm1 = read[t]
        h_tm1 = h_t
        if t < decoderSize then
            P[t+1] = P_t - phi_t
        end

    end

    self.output = output
    return self.output
end

function PriorityQueueSimpleDecoderV2:updateGradInput(input, gradOutput)

    local M, P_0, Y = unpack(input)
    local queueSize = M:size(1)
    local decoderSize = Y:size(1)
    local dimSize = self.inputSize
    local batchSize = Y:size(2)

    local read_0 = self.read_0

    local read = self.read
    local gradM = self.gradM:resizeAs(M):zero()
    local gradY = self.gradY:resizeAs(Y):zero()
    local h_0 = self.h_0:expand(batchSize, dimSize)

    local P = self.P
    local rho = self.rho
    local rhocum = self.rhocum
    local phicum = self.phicum

    if self.prevQueueSize < queueSize then
       
        if string.match(Y:type(), "Cuda") then
            self.lt_mask_base = self.lt_mask_base:type("torch.CudaByteTensor")
            self.di_mask_base = self.di_mask_base:type("torch.CudaByteTensor")
        else
            self.lt_mask_base = self.lt_mask_base:type("torch.ByteTensor")
            self.di_mask_base = self.di_mask_base:type("torch.ByteTensor")
        end

        self.lt_mask_base:resize(queueSize, queueSize):fill(1.0)
        torch.tril(self.lt_mask_base, self.lt_mask_base, -1)

        self.di_mask_base:resize(queueSize, queueSize):zero()

        self.grad_wrt_pi_vals_base:resize(queueSize, queueSize):zero()

        self.di_mask_base[1][1] = 1
        self.grad_wrt_pi_vals_base[1][1] = 1

        for i=2,queueSize do 
            self.di_mask_base[i][i] = 1 
            self.grad_wrt_pi_vals_base[i][i] = 1
            self.grad_wrt_pi_vals_base[i][{{1,i-1}}] = -1
        end

        self.prevQueueSize = queueSize
    end


    local lt_mask = self.lt_mask_base:view(
        1, 1, self.prevQueueSize, self.prevQueueSize)[
            {{}, {}, {1, queueSize}, {1, queueSize}}]:expand(
                decoderSize, batchSize, queueSize, queueSize)

    local di_mask = self.di_mask_base:view(
        1,1, self.prevQueueSize, self.prevQueueSize)[
            {{}, {}, {1, queueSize}, {1, queueSize}}]:expand(
                decoderSize, batchSize, queueSize, queueSize)

    local grad_wrt_P_vals = 
        self.grad_wrt_pi_vals_base:view(
            1, 1, self.prevQueueSize, self.prevQueueSize)[
                {{}, {}, {1, queueSize}, {1, queueSize}}]:expand(
                    decoderSize, batchSize, queueSize, queueSize)


    self.maskBuffer1 = self.maskBuffer1:typeAs(lt_mask)
    self.maskBuffer2 = self.maskBuffer2:typeAs(lt_mask)
    local rho_mask_base = torch.gt(P, rhocum):cmul(torch.gt(rhocum, 0))
    local phi_mask_base = torch.gt(P, phicum):cmul(torch.gt(phicum, 0))

    local grad_rho_wrt_r = self.grad_rho_wrt_r:resize(
        decoderSize, batchSize, queueSize):copy(rho_mask_base)
    local grad_phi_wrt_f = self.grad_phi_wrt_f:resize(
        decoderSize, batchSize, queueSize):copy(phi_mask_base)

    local rho_P_lt_mask = torch.cmul(
        self.maskBuffer1,
        rho_mask_base:view(
            decoderSize, batchSize, queueSize, 1):expand(
            decoderSize, batchSize, queueSize, queueSize),
        lt_mask)
    local rho_P_d_mask = torch.cmul(
        self.maskBuffer2,
        torch.lt(P, rhocum):cmul(torch.gt(P, 0)):view(
            decoderSize, batchSize, queueSize, 1):expand(
            decoderSize, batchSize, queueSize, queueSize),
        di_mask)
    local rho_P_mask = rho_P_lt_mask:add(rho_P_d_mask)
    self.maskBuffer3:resize(decoderSize, batchSize, queueSize, queueSize)
    local grad_rho_wrt_P = self.maskBuffer3:copy(rho_P_mask):cmul(
        grad_wrt_P_vals)
    
    local phi_P_lt_mask = torch.cmul(
        self.maskBuffer1,
        phi_mask_base:view(
            decoderSize, batchSize, queueSize, 1):expand(
            decoderSize, batchSize, queueSize, queueSize),
        lt_mask)
    local phi_P_d_mask = torch.cmul(
        self.maskBuffer2,
        torch.lt(P, phicum):cmul(torch.gt(P, 0)):view(
            decoderSize, batchSize, queueSize, 1):expand(
            decoderSize, batchSize, queueSize, queueSize),
        di_mask)
    local phi_P_mask = phi_P_lt_mask:add(phi_P_d_mask)
    self.maskBuffer4:resize(decoderSize, batchSize, queueSize, queueSize)
    local grad_phi_wrt_P = self.maskBuffer4:copy(phi_P_mask):cmul(
        grad_wrt_P_vals)
    
    local h_tm1 
    local read_tm1

    local grad_P_tp1 = self.gradP:resize(batchSize, queueSize):zero()


    if self.isMasked then
        gradOutput:maskedFill(self.Ymask, 0)
        self.Pmask = torch.eq(M[{{1,queueSize},{1, batchSize}, {1}}], 0)
        self.Pmask = self.Pmask:view(queueSize, batchSize):t()
    end

    for t=decoderSize,1,-1 do

        if t > 1 then
            local controller_tm1 = self:getController(t-1)
            h_tm1 = controller_tm1.output
            read_tm1 = read[t-1]
        else
            h_tm1 = h_0
            read_tm1 = read_0:view(batchSize, dimSize)
        end

        local controller_t = self:getController(t)
        local h_t = controller_t.output
        controller_input_t = {h_tm1, read_tm1, Y[t]}

        if t < decoderSize then
            local controller_tp1 = self:getController(t + 1)
            local grad_h_t = controller_tp1.gradInput[1]
            local grad_read_t = controller_tp1.gradInput[2]

            gradM:add(
                torch.cmul(
                    self.buffer1,
                    grad_read_t:view(batchSize, 1, dimSize):expand(
                        batchSize, queueSize, dimSize),
                    rho[t]:view(batchSize, queueSize, 1):expand(
                        batchSize, queueSize, dimSize)):permute(2, 1, 3))

            local grad_h_tp1_wrt_read_t = 
                torch.sum(
                    self.grad_h_tp1_wrt_read_t,
                    torch.cmul(
                        self.buffer1, 
                        grad_read_t:view(batchSize, 1, dimSize):expand(
                            batchSize, queueSize, dimSize),
                        M:permute(2, 1, 3)),
                3)

            local grad_h_tp1_wrt_r = torch.sum(
                self.grad_h_tp1_wrt_r,
                torch.cmul(
                    self.buffer1,
                    grad_h_tp1_wrt_read_t:view(batchSize, queueSize),
                    grad_rho_wrt_r[t]),
                2)
            
            local grad_P_tp1_wrt_f = torch.sum(
                self.grad_P_tp1_wrt_f,
                torch.cmul(
                    self.buffer1,
                    torch.mul(
                        self.buffer2,
                        grad_P_tp1, 
                        -1),
                    grad_phi_wrt_f[t]),
                2)
            
            local grad_P_tp1_wrt_P_t = torch.bmm(
                self.buffer1:resize(batchSize, 1, queueSize),
                self.buffer2:view(batchSize, 1, queueSize),
                grad_phi_wrt_P[t]):view(batchSize, queueSize)

            grad_P_tp1:add(grad_P_tp1_wrt_P_t)

            grad_P_tp1:add(
                torch.bmm(
                    self.buffer1,
                    grad_h_tp1_wrt_read_t:permute(1,3,2),
                    grad_rho_wrt_P[t]):view(batchSize, queueSize))

            if self.isMasked then
                grad_P_tp1:maskedFill(self.Pmask, 0)
            end

            local grad_h_tp1_wrt_h_t = self:getReadLayer(t):backward(
                h_t, grad_h_tp1_wrt_r)
            local grad_P_tp1_wrt_h_t = self:getForgetLayer(t):backward(
                h_t, grad_P_tp1_wrt_f)
            local grad_controller_t_out = grad_h_tp1_wrt_h_t:add(
                grad_P_tp1_wrt_h_t):add(grad_h_t):add(gradOutput[t])
            
            controller_t:backward(controller_input_t, grad_controller_t_out)
        else
            controller_t:backward(controller_input_t, gradOutput[t]) 
        end

        gradY[t] = controller_t.gradInput[3]
        
    end

    torch.sum(
        self.grad_h_0,
        self:getController(1).gradInput[1],
        1)

    local gradP0 = grad_P_tp1:t():add(
        torch.sum(
            self.buffer1,
            torch.cmul(
                self.buffer2,
                M, 
                self:getController(1).gradInput[2]:view(
                    1, batchSize, dimSize):expand(
                    queueSize, batchSize, dimSize)),
            3):view(queueSize, batchSize))

    gradM:add(
        torch.cmul(
            self.buffer1,
            self:getController(1).gradInput[2]:view(
                1, batchSize, dimSize):expand(
                queueSize, batchSize, dimSize),
            P_0:view(queueSize, batchSize, 1):expand(
                queueSize, batchSize, dimSize)))

    self.gradInput = {gradM, gradP0, gradY}
    return self.gradInput
end
