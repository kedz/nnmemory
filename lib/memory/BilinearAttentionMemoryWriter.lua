local BilinearAttentionMemoryWriter, Parent = 
    torch.class('nn.BilinearAttentionMemoryWriter', 'nn.Module')

function BilinearAttentionMemoryWriter:__init(dimSize)

    self.dimSize = dimSize
    --self.linear = nn.BilinearAttention(dimSize, 1)
    self.output = torch.Tensor()
    self.gradInput = torch.Tensor()
    self.buffer = torch.Tensor()
    self.gradBuffer = torch.Tensor()

end

function BilinearAttentionMemoryWriter:updateOutput(input)
    local dimSize = self.dimSize
    assert(input:dim() == 3)
    assert(input:size(3) == dimSize)
    local batchSize = input:size(2)
    local maxSteps = input:size(1)
    local output = self.output:resize(maxSteps, batchSize, 1)
    output = output:view(maxSteps, batchSize, 1, 1)

    local X = input:view(maxSteps, batchSize, 1, dimSize) 
    local Xcum = self.buffer:resizeAs(X[1]):zero()
    for t=maxSteps-1,1,-1 do
        Xcum:add(X[t+1])
        torch.bmm(output[t], X[t], Xcum:permute(1,3,2))
    end
    return self.output
end

function BilinearAttentionMemoryWriter:updateGradInput(input, gradOutput)
    local dimSize = self.dimSize
    assert(input:dim() == 3)
    assert(input:size(3) == dimSize)
    local batchSize = input:size(2)
    local maxSteps = input:size(1)
    -- TODO  add some asserts for gradOutput


    local grad_o_wrt_x = torch.cmul(
        self.gradBuffer,
        gradOutput:expand(maxSteps, batchSize, dimSize),
        input):view(maxSteps, 1, batchSize, dimSize)

    local X = input:view(maxSteps, batchSize, 1, dimSize) 
    local Xcum = self.buffer:resizeAs(X[1]):zero()
    local XcumT = Xcum:transpose(3,2)
    local gradInput = 
        self.gradInput:resize(maxSteps, batchSize, dimSize):zero()
    gradInput = gradInput:view(maxSteps, batchSize, dimSize, 1)
    for t=maxSteps-1,1,-1 do
        Xcum:add(X[t+1])
        torch.cmul(
            gradInput[t],
            XcumT:view(batchSize, dimSize),
            gradOutput[t]:expand(batchSize, dimSize))
        self.gradInput[{{t+1,maxSteps},{},{}}]:add( 
            grad_o_wrt_x[t]:expand(maxSteps - t, batchSize, dimSize))
    end
    return self.gradInput
end

