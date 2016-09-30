local LinearAssociativeMemoryReader, Parent =
    torch.class('nn.LinearAssociativeMemoryReader', 'nn.Module')

function LinearAssociativeMemoryReader:__init(dimSize)

    self.dimSize = dimSize
    self.isMaskZero = false

    self.weightMemory = torch.Tensor(dimSize, dimSize)
    self.weightInput = torch.Tensor(dimSize, dimSize)
    self.weightAttention = torch.Tensor(dimSize)
    self.bias = torch.Tensor(dimSize)

    self.gradWeightMemory = torch.Tensor(dimSize, dimSize)
    self.gradWeightInput = torch.Tensor(dimSize, dimSize)
    self.gradWeightAttention = torch.Tensor(dimSize)
    self.gradBias = torch.Tensor(dimSize)

    self.output = torch.Tensor()

    self.buffer1 = torch.Tensor()
    self.buffer2 = torch.Tensor()
    self.buffer3 = torch.Tensor()

    self.gradY = torch.Tensor()
    self.gradM = torch.Tensor()

    self.attNetsPreMask = {}
    self.attNetsPostMask = {}

    self.attActNet = nn.Sequential():add(
            nn.CAddTable()):add(
            nn.Tanh())

    self:reset()
end

function LinearAssociativeMemoryReader:maskZero()
    self.isMaskZero = true
    return self
end

function LinearAssociativeMemoryReader:reset()
    self.weightMemory:uniform(-1,1)
    self.weightInput:uniform(-1,1)
    self.bias:fill(0)
    self.weightAttention:uniform(-1,1)
    self:zeroGradParameters()
end

function LinearAssociativeMemoryReader:zeroGradParameters()
    self.gradWeightMemory:zero()
    self.gradWeightInput:zero()
    self.gradWeightAttention:zero()
    self.gradBias:zero()
end

function LinearAssociativeMemoryReader:parameters()
    local params = {self.weightMemory,
                    self.weightInput,
                    self.weightAttention,
                    self.bias}
    local gradParams = {self.gradWeightMemory,
                        self.gradWeightInput,
                        self.gradWeightAttention,
                        self.gradBias}
    return params, gradParams
end

function LinearAssociativeMemoryReader:updateOutput(input)
    local M, Y = unpack(input)
    local dimSize = self.dimSize

    assert(M:dim() == 3)
    assert(M:size(3) == dimSize)
    assert(Y:dim() == 3)
    assert(Y:size(3) == dimSize)
    local memorySize = M:size(1)
    local batchSize = Y:size(2)
    local maxSteps = Y:size(1)
    local output = self.output:resize(maxSteps, batchSize, dimSize)

    local MT = M:transpose(2,1)

    local MTmask
    local Ymask 
    if self.isMaskZero then
        MTmask = torch.eq(MT:select(3,1), 0)
        Ymask = torch.eq(Y, 0)
    end

    local Wmem = self.weightMemory:view(1, dimSize, dimSize):expand(
        batchSize, dimSize, dimSize)
    local Winp = self.weightInput:view(1, dimSize, dimSize):expand(
        batchSize, dimSize, dimSize)
    local Wattn = self.weightAttention:view(1,dimSize, 1):expand(
        batchSize, dimSize, 1)
    local bias = self.bias:view(1, 1, 1, dimSize):expand(
       maxSteps, batchSize, memorySize, dimSize)
   


    if self.prevMemorySize ~= memorySize then
        self.inpActNet = nn.Sequential():add(
            nn.ParallelTable():add(
                nn.Transpose({2,1})):add(
                nn.Identity())):add(
            nn.MM()):add(
            nn.Transpose({2,1})):add(
            nn.Replicate(memorySize, 3))
        self.inpActNet:type(self:type())
        self.prevMemorySize = memorySize
    end

    if self.prevMaxSteps ~= maxSteps then
        self.memActNet = nn.Sequential():add(
            nn.ParallelTable():add(
                nn.Transpose({2,1})):add(
                nn.Identity())):add(
            nn.MM()):add(
            nn.Replicate(maxSteps))
        self.memActNet:type(self:type())
        self.prevMaxSteps = maxSteps
    end

    self.inputAct = self.inpActNet:forward({Y, Winp})
    self.memoryAct = self.memActNet:forward({M, Wmem})



    self.attAct = self.attActNet:forward(
        {self.inputAct, self.memoryAct, bias})

    for t=1,maxSteps do
        local attNetPreMask = self.attNetsPreMask[t]
        local attNetPostMask = self.attNetsPostMask[t]
        if attNetPreMask == nil then
            self.attNetsPreMask[t] = nn.Sequential():add(
                nn.MM()):add(nn.Squeeze(3))
            self.attNetsPreMask[t]:type(self:type())
                
            attNetPreMask = self.attNetsPreMask[t]
            self.attNetsPostMask[t] = nn.SoftMax()
            self.attNetsPostMask[t]:type(self:type())
            attNetPostMask = self.attNetsPostMask[t]
        end

        local attPreMask = attNetPreMask:forward({self.attAct[t], Wattn})
        if self.isMaskZero then
            attPreMask[MTmask] = -math.huge
        end
        local att = attNetPostMask:forward(attPreMask)

        att = att:view(batchSize, memorySize, 1):expand(
            batchSize, memorySize, dimSize)
        local memoryWeighted = torch.cmul(self.buffer1, att, MT)
        
        output[t]:view(batchSize, 1, dimSize):sum(memoryWeighted, 2)
    end

    if self.isMaskZero then
        self.output[Ymask] = 0
    end
    return self.output
end

function LinearAssociativeMemoryReader:updateGradInput(input, gradOutput)

    local M, Y = unpack(input)
    local memorySize = M:size(1)
    local maxSteps = Y:size(1)
    local batchSize = Y:size(2)
    local dimSize = self.dimSize

    local MT = M:transpose(2,1)
    local YT = Y:transpose(2,1)

    local Wmem = self.weightMemory:view(1, dimSize, dimSize):expand(
        batchSize, dimSize, dimSize)
    local Winp = self.weightInput:view(1, dimSize, dimSize):expand(
        batchSize, dimSize, dimSize)
    local Wattn = self.weightAttention:view(1,dimSize, 1):expand(
        batchSize, dimSize, 1)
    local bias = self.bias:view(1, 1, 1, dimSize):expand(
       maxSteps, batchSize, memorySize, dimSize)
 
    local grad_c_wrt_att = MT

    local gradAttAll = self.buffer1:resizeAs(self.attAct):zero()
    local gradY = self.gradY:resizeAs(Y):zero()
    local gradM = self.gradM:resizeAs(M):zero()

    for t=maxSteps,1,-1 do
        local gradOutput_t = gradOutput[t]:view(batchSize, 1, dimSize):expand(
            batchSize, memorySize, dimSize)
        local att = self.attNetsPostMask[t].output
        gradM:add(torch.cmul(
                self.buffer2,
                gradOutput_t, 
                att:view(batchSize, memorySize, 1):expand(
                batchSize, memorySize, dimSize)):transpose(2,1))
            
        local grad_o_wrt_att = torch.cmul(
            self.buffer2,
            gradOutput_t, 
            grad_c_wrt_att)
        grad_o_wrt_att = torch.sum(
            self.buffer3,
            grad_o_wrt_att,
            3):view(batchSize, memorySize)

        local gradAttPost = self.attNetsPostMask[t]:backward(
            self.attNetsPreMask[t].output, grad_o_wrt_att)
        local gradAtt = self.attNetsPreMask[t]:backward(
            {self.attAct[t], Wattn}, gradAttPost)
        gradAttAll[t] = gradAtt[1]
        self.gradWeightAttention:add(
            torch.sum(
                self.buffer2,
                gradAtt[2],
                1))
        
    end

    local gradAtt = self.attActNet:backward(
        {self.inputAct, self.memoryAct, bias}, 
        gradAttAll)

    local gradInputAct = self.inpActNet:backward({YT, Winp}, gradAtt[1])
    local gradMemoryAct = self.memActNet:backward({MT, Wmem}, gradAtt[2])

    gradY:add(gradInputAct[1])
    self.gradWeightInput:add(torch.sum(self.buffer2, gradInputAct[2], 1))
    self.gradWeightMemory:add(torch.sum(self.buffer2, gradMemoryAct[2], 1))
    gradM:add(gradMemoryAct[1])
    self.gradInput = {gradM, gradY}

    self.gradBias:add(
        torch.sum(
            self.buffer2,
            torch.sum(
                self.buffer3,
                torch.sum(
                    self.buffer2,
                    gradAtt[3],
                    1),
                2),
            3))

    return self.gradInput

end
