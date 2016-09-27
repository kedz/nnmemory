local MemoryCell, Parent = torch.class('nn.MemoryCell', 'nn.Module')

function MemoryCell:__init()

    self.isMaskZero = false
    self.writers = nn.Concat(3)
    self.net = nn.Sequential():add(self.writers):add(nn.Sum(3, 3, false))

    self.gradInput = torch.Tensor()
    self.sm = nn.Sequential()
    self.sm:add(nn.Transpose({2, 1}))
    self.sm:add(nn.SoftMax())
    self.sm:add(nn.Transpose({1, 2}))
    self:zeroGradParameters()
end

function MemoryCell:maskZero()
    self.isMaskZero = true
    return self
end

function MemoryCell:add(writer)
    self.writers:add(writer)
    return self
end

function MemoryCell:updateOutput(input)
    self.activation = self.net:forward(input)
    if self.isMaskZero then
        self.activation[torch.eq(input[{{},{},{1,1}}], 0)] = -math.huge
    end
    self.activation_normalized = self.sm:forward(self.activation)
    self.output = {input, self.activation_normalized}
    return self.output
end

function MemoryCell:updateGradInput(input, gradOutput)
    local gradSoftMax = self.sm:updateGradInput(self.activation, gradOutput[2])
    local gradWriter = self.net:updateGradInput(input, gradSoftMax)
    torch.add(self.gradInput, gradWriter, gradOutput[1]) 
    return self.gradInput
end

function MemoryCell:accGradParameters(input, gradOutput)
    self.net:accGradParameters(input, self.sm.gradInput)
end

function MemoryCell:parameters()
    return self.net:parameters()
end

function MemoryCell:zeroGradParameters()
    self.net:zeroGradParameters()
end
