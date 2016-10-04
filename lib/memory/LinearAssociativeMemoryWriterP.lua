local LinearAssociativeMemoryWriterP, Parent = 
    torch.class('nn.LinearAssociativeMemoryWriterP', 'nn.Module')


function LinearAssociativeMemoryWriterP:__init(dimSize, mode)
    self.mode = mode or "forward"

    assert(self.mode == "forward" or self.mode == "backward" or 
        self.mode == "all")

    self.dimSize = dimSize
    self.isMaskZero = false

    self.weightMemory = torch.Tensor(dimSize, dimSize)
    self.weightInput = torch.Tensor(dimSize, dimSize)
    self.weightAttention = torch.Tensor(dimSize)
 --   self.bias = torch.Tensor(dimSize)

    self.gradWeightMemory = torch.Tensor(dimSize, dimSize)
    self.gradWeightInput = torch.Tensor(dimSize, dimSize)
    self.gradWeightAttention = torch.Tensor(dimSize)
--    self.gradBias = torch.Tensor(dimSize)
    self.buffer = torch.Tensor()

    self.gradInput = torch.Tensor()

    self:reset()

end

function LinearAssociativeMemoryWriterP:updateOutput(input)

    local M = input
    local dimSize = self.dimSize

    assert(M:dim() == 3)
    assert(M:size(3) == dimSize)
    local batchSize = M:size(2)
    local maxSteps = M:size(1)

    if self.isMaskZero then
        self.mask = torch.eq(M, 0)
    end

    if self.prevMaxSteps ~= maxSteps then
        local maskCurrentTime = torch.eye(maxSteps)
        maskCurrentTime = torch.ne(
            maskCurrentTime:view(maxSteps,1,maxSteps,1):expand(
                maxSteps,batchSize,maxSteps,dimSize), 1)
        maskCurrentTime = maskCurrentTime:typeAs(input)
        self.maskCurrentTime = maskCurrentTime:contiguous()
        if self.mode == "forward" then
            for i=2,maxSteps do
                self.maskCurrentTime:select(1, i)[{{},{1,i},{}}]:fill(0)
            end
        elseif self.mode == "backward" then
            for i=maxSteps-1,1,-1 do
                self.maskCurrentTime:select(1, i)[{{},{i,maxSteps},{}}]:fill(0)
            end
        end
    end

    local Wmem = self.weightMemory:view(1, dimSize, dimSize):expand(
        batchSize, dimSize, dimSize)
    local Winp = self.weightInput:view(1, dimSize, dimSize):expand(
        batchSize, dimSize, dimSize)
    local Wattn = self.weightAttention:view(1,dimSize, 1):expand(
        maxSteps, dimSize, 1)
    --local bias = self.bias:view(1, 1, dimSize):expand(
    --   maxSteps, batchSize, dimSize)
 
    if self.prevMaxSteps ~= maxSteps then
        self.inpActNet = nn.Sequential():add(
            nn.ParallelTable():add(
                nn.Transpose({2,1})):add(
                nn.Identity())):add(
            nn.MM()):add(
            nn.Transpose({2,1}))--:add(
            --nn.Replicate(maxSteps, 3))

        self.memActNet = nn.Sequential():add(
            nn.ParallelTable():add(
                nn.Sequential():add(
                    nn.ParallelTable():add(
                        nn.Transpose({2,1})):add(
                        nn.Identity())):add(
                    nn.MM()):add(
                    nn.Replicate(maxSteps))):add(
                nn.Identity())):add(
            nn.CMulTable()):add(
            nn.Sum(3, false))
        
        self.net = nn.Sequential():add(
            nn.ParallelTable():add(
                nn.Sequential():add(
                    nn.ParallelTable():add(
                        self.inpActNet):add(
                        self.memActNet)):add(
                    nn.CAddTable()):add(
                    nn.Tanh())):add(
                nn.Identity())):add(
            nn.MM())

        self.net:type(self:type())
        self.prevMaxSteps = maxSteps
    end

    self.output = self.net:forward(
        {{{M, Winp}, {{M, Wmem}, self.maskCurrentTime}}, Wattn})

    return self.output

end

function LinearAssociativeMemoryWriterP:updateGradInput(input, gradOutput)
    local M = input
    local dimSize = self.dimSize

    assert(M:dim() == 3)
    assert(M:size(3) == dimSize)
    local batchSize = M:size(2)
    local maxSteps = M:size(1)

    local grads = self.net:backward(
        {{{M, Winp}, {{M, Wmem}, self.maskCurrentTime}}, Wattn}, 
        gradOutput)
    self.gradWeightAttention:add(torch.sum(self.buffer, grads[2], 1)) 
    self.gradWeightInput:add(
        torch.sum(self.buffer, grads[1][1][2], 1))
    self.gradWeightMemory:add(
        torch.sum(self.buffer, grads[1][2][1][2], 1))
    --self.gradWeightBias:add(grads[1][1][2][2])

    torch.add(self.gradInput, grads[1][1][1], grads[1][2][1][1])
    return self.gradInput

end

function LinearAssociativeMemoryWriterP:reset()
    self.weightMemory:uniform(-1,1)
    self.weightInput:uniform(-1,1)
    --self.bias:fill(0)
    self.weightAttention:uniform(-1,1)
    self:zeroGradParameters()
end

function LinearAssociativeMemoryWriterP:zeroGradParameters()
    self.gradWeightMemory:zero()
    self.gradWeightInput:zero()
    self.gradWeightAttention:zero()
  --  self.gradBias:zero()
end

function LinearAssociativeMemoryWriterP:parameters()
    local params = {self.weightMemory,
                    self.weightInput,
                    self.weightAttention
                   }
                   -- self.bias}
    local gradParams = {self.gradWeightMemory,
                        self.gradWeightInput,
                        self.gradWeightAttention
                       }
                     --   self.gradBias}
    return params, gradParams
end
