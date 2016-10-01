local LinearAssociativeMemoryWriter, Parent = 
    torch.class('nn.LinearAssociativeMemoryWriter', 'nn.Module')


function LinearAssociativeMemoryWriter:__init(dimSize, mode)
    self.mode = mode or "forward"

    assert(self.mode == "forward" or self.mode == "backward" or 
        self.mode == "all")

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

    self.gradM = torch.Tensor()

    self.attNetsPreMask = {}
    self.attNetsPostMask = {}

    self.attActNet = nn.Sequential():add(
            nn.CAddTable()):add(
            nn.Tanh())


    self:reset()

end

function LinearAssociativeMemoryWriter:updateOutput(input)

    local M = input
    local dimSize = self.dimSize

    assert(M:dim() == 3)
    assert(M:size(3) == dimSize)
    local batchSize = M:size(2)
    local maxSteps = M:size(1)
    local output = self.output:resize(maxSteps, batchSize, dimSize)

    if self.prevMaxSteps ~= maxSteps then
        local maskCurrentTime = torch.eye(maxSteps)
        maskCurrentTime = torch.ne(
            maskCurrentTime:view(maxSteps,1,maxSteps,1):expand(
                maxSteps,batchSize,maxSteps,dimSize), 1)
        maskCurrentTime = maskCurrentTime:typeAs(input)
        self.maskCurrentTime = maskCurrentTime
    end

    local MT = M:transpose(2,1)

    local MTmask
    local Mmask 
    if self.isMaskZero then
        MTmask = torch.eq(MT:select(3,1), 0)
        Mmask = torch.eq(M:select(3,1), 0):view(maxSteps,batchSize,1):expand(
            maxSteps, batchSize, maxSteps)
    end

    local Wmem = self.weightMemory:view(1, dimSize, dimSize):expand(
        batchSize, dimSize, dimSize)
    local Winp = self.weightInput:view(1, dimSize, dimSize):expand(
        batchSize, dimSize, dimSize)
    local Wattn = self.weightAttention:view(1,dimSize, 1):expand(
        batchSize, dimSize, 1)
    local bias = self.bias:view(1, 1, 1, dimSize):expand(
       maxSteps, batchSize, maxSteps, dimSize)
 
    if self.prevMaxSteps ~= maxSteps then
        self.inpActNet = nn.Sequential():add(
            nn.ParallelTable():add(
                nn.Transpose({2,1})):add(
                nn.Identity())):add(
            nn.MM()):add(
            nn.Transpose({2,1})):add(
            nn.Replicate(maxSteps, 3))
        self.inpActNet:type(self:type())
        self.memActNet = nn.Sequential():add(
            nn.ParallelTable():add(
                nn.Sequential():add(
                    nn.ParallelTable():add(
                        nn.Transpose({2,1})):add(
                        nn.Identity())):add(
                    nn.MM()):add(
                    nn.Replicate(maxSteps))):add(
                nn.Identity())):add(
            nn.CMulTable())
        self.memActNet:type(self:type())
        self.prevMaxSteps = maxSteps
    end

    self.inputAct = self.inpActNet:forward({M, Winp})
    self.memoryAct = self.memActNet:forward({{M, Wmem}, maskCurrentTime})

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
        if self.mode == "forward" and t > 1 then
            attPreMask[{{},{1,t-1}}]:fill(-math.huge)
        elseif self.mode == "backward" and t < maxSteps then
            attPreMask[{{},{t+1,maxSteps}}]:fill(-math.huge)
        end

        if self.isMaskZero then
            attPreMask[MTmask] = -math.huge
        end
        local att = attNetPostMask:forward(attPreMask)
        
        if self.isMaskZero then
            att[Mmask[t]] = 0
        end

        att = att:view(batchSize, maxSteps, 1):expand(
            batchSize, maxSteps, dimSize)
        local memoryWeighted = torch.cmul(self.buffer1, att, MT)
        
        output[t]:view(batchSize, 1, dimSize):sum(memoryWeighted, 2)
    end

    return self.output

end

function LinearAssociativeMemoryWriter:updateGradInput(input, gradOutput)

    local M = input
    local maxSteps = M:size(1)
    local batchSize = M:size(2)
    local dimSize = self.dimSize

    local MT = M:transpose(2,1)

    local Wmem = self.weightMemory:view(1, dimSize, dimSize):expand(
        batchSize, dimSize, dimSize)
    local Winp = self.weightInput:view(1, dimSize, dimSize):expand(
        batchSize, dimSize, dimSize)
    local Wattn = self.weightAttention:view(1,dimSize, 1):expand(
        batchSize, dimSize, 1)
    local bias = self.bias:view(1, 1, 1, dimSize):expand(
       maxSteps, batchSize, maxSteps, dimSize)
 
    local grad_c_wrt_att = MT

    local gradAttAll = self.buffer1:resizeAs(self.attAct):zero()
    local gradM = self.gradM:resizeAs(M):zero()

    for t=maxSteps,1,-1 do
        local gradOutput_t = gradOutput[t]:view(batchSize, 1, dimSize):expand(
            batchSize, maxSteps, dimSize)
        local att = self.attNetsPostMask[t].output
        gradM:add(torch.cmul(
                self.buffer2,
                gradOutput_t, 
                att:view(batchSize, maxSteps, 1):expand(
                batchSize, maxSteps, dimSize)):transpose(2,1))
            
        local grad_o_wrt_att = torch.cmul(
            self.buffer2,
            gradOutput_t, 
            grad_c_wrt_att)
        grad_o_wrt_att = torch.sum(
            self.buffer3,
            grad_o_wrt_att,
            3):view(batchSize, maxSteps)

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

    local gradInputAct = self.inpActNet:backward({MT, Winp}, gradAtt[1])
    local gradMemoryAct = self.memActNet:backward(
        {{MT, Wmem}, self.maskCurrentTime}, gradAtt[2])[1]

    gradM:add(gradInputAct[1])
    self.gradWeightInput:add(torch.sum(self.buffer2, gradInputAct[2], 1))
    self.gradWeightMemory:add(torch.sum(self.buffer2, gradMemoryAct[2], 1))
    gradM:add(gradMemoryAct[1])
    self.gradInput = gradM

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

function LinearAssociativeMemoryWriter:maskZero()
    self.isMaskZero = true
    return self
end

function LinearAssociativeMemoryWriter:reset()
    self.weightMemory:uniform(-1,1)
    self.weightInput:uniform(-1,1)
    self.bias:fill(0)
    self.weightAttention:uniform(-1,1)
    self:zeroGradParameters()
end

function LinearAssociativeMemoryWriter:zeroGradParameters()
    self.gradWeightMemory:zero()
    self.gradWeightInput:zero()
    self.gradWeightAttention:zero()
    self.gradBias:zero()
end

function LinearAssociativeMemoryWriter:parameters()
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



