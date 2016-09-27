local LinearMemoryWriter, Parent = 
    torch.class('nn.LinearMemoryWriter', 'nn.Module')

function LinearMemoryWriter:__init(dimSize)

    self.dimSize = dimSize
    self.linear = nn.Linear(dimSize, 1)
    self.output = torch.Tensor()
    self.gradInput = torch.Tensor()

end

function LinearMemoryWriter:parameters()
    return self.linear:parameters()
end

function LinearMemoryWriter:updateOutput(input)
    local dimSize = self.dimSize
    assert(input:dim() == 3)
    assert(input:size(3) == dimSize)
    local batchSize = input:size(2)
    local maxSteps = input:size(1)
    local output = self.output:resize(maxSteps, batchSize, 1)

    for t=1,maxSteps do
        output[t] = self.linear:forward(input[t])
    end
    return self.output
end

function LinearMemoryWriter:updateGradInput(input, gradOutput)
    local dimSize = self.dimSize
    assert(input:dim() == 3)
    assert(input:size(3) == dimSize)
    local batchSize = input:size(2)
    local maxSteps = input:size(1)
    -- TODO  add some asserts for gradOutput
    
    local gradInput = self.gradInput:resize(maxSteps, batchSize, dimSize)
    for t=maxSteps,1,-1 do
        self.gradInput[t] = 
            self.linear:updateGradInput(input[t], gradOutput[t])
    end
    return gradInput
end

function LinearMemoryWriter:accGradParameters(input, gradOutput)
    local dimSize = self.dimSize
    assert(input:dim() == 3)
    assert(input:size(3) == dimSize)
    local batchSize = input:size(2)
    local maxSteps = input:size(1)
    -- TODO  add some asserts for gradOutput
    
    for t=maxSteps,1,-1 do
        self.linear:accGradParameters(input[t], gradOutput[t])
    end
end

function LinearMemoryWriter:zeroGradParameters()
    self.linear:zeroGradParameters()
end
