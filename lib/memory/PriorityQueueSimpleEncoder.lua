local PriorityQueueSimpleEncoder, Parent = 
    torch.class('nn.PriorityQueueSimpleEncoder', 'nn.Module')

function PriorityQueueSimpleEncoder:__init(dimSize)
    Parent.__init(self)
    self.dimSize = dimSize

    self.weight = torch.Tensor(dimSize)    
    self.gradWeight = torch.Tensor(dimSize)    
    self.bias = torch.Tensor(1)    
    self.gradBias = torch.Tensor(1)    

    self.isMaskZero = false
    
    self:reset()

end

function PriorityQueueSimpleEncoder:maskZero()
    self.isMaskZero = true
    return self
end

function PriorityQueueSimpleEncoder:reset()
    self.weight:uniform(-1, 1)
    self.bias:fill(1) -- initial bias is to write
    self:zeroGradParameters()
end

function PriorityQueueSimpleEncoder:zeroGradParameters()
    self.gradWeight:zero()
    self.gradBias:zero()
end

function PriorityQueueSimpleEncoder:updateOutput(input)
    assert(input:dim() == 3, 
        "Input must be a batch of dim seqLength x batch size x dim size")
    assert(input:size(3) == self.dimSize, 
        "Input must have dimension dimSize")

    local d = self.dimSize
    local maxSteps = input:size(1)
    local batchSize = input:size(2)

    local W = self.weight:view(1,d,1):expand(batchSize, d, 1)
    local b = self.bias:view(1,1,1):expand(batchSize, 1, 1)
    local pi = torch.Tensor():resize(maxSteps, batchSize, 1):typeAs(b):zero()

    local X = input:view(maxSteps, batchSize, 1, d)

    for t=1,maxSteps do

        local pi_t = pi[t]:view(batchSize, 1, 1) 
        torch.baddbmm(pi_t, b, X[t], W)
    end

    pi = pi:view(maxSteps,batchSize)

    if self.isMaskZero then
        local mask = torch.eq(input[{{},{},{1}}], 0):nonzero()
        if mask:dim() > 0 then 
            mask = mask[{{},{1,2}}]
            for m=1,mask:size(1) do
                pi[mask[m][1]][mask[m][2]] = -math.huge
            end
        end
    end

    torch.exp(pi, pi)
    torch.cdiv(pi, pi, torch.sum(pi, 1):expand(maxSteps,batchSize))
    
    local pi_sorted, I = torch.sort(pi, 1, true)
    local memory_sorted = torch.Tensor():resizeAs(input):typeAs(input):zero()

    for t=1,maxSteps do
        for b=1,batchSize do
            memory_sorted[t][b]:copy(input[I[t][b]][b])    
        end
    end

    self.indices = I
    self.output = {memory_sorted, pi_sorted}
    return self.output
end

function PriorityQueueSimpleEncoder:updateGradInput(input, gradOutput)

    local d = self.dimSize

    local grad_M = gradOutput[1]
    local M = self.output[1]
    local pi = self.output[2]

    local maxSteps = input:size(1)
    local batchSize = input:size(2)
    local W = self.weight:view(1,1,d):expand(batchSize,1,d)
    local I = 
        torch.eye(maxSteps):view(maxSteps, maxSteps, 1):expand(
            maxSteps, maxSteps, batchSize)
    local J = torch.cmul(
        pi:view(maxSteps,1,batchSize):expand(maxSteps,maxSteps,batchSize),
        I - pi:view(1,maxSteps,batchSize):expand(maxSteps,maxSteps,batchSize))
    J = J:permute(3,1,2)
    local grad_pi = gradOutput[2]:permute(2,1):contiguous()
    local grad_pi = grad_pi:view(batchSize,maxSteps,1)
    local grad_o_wrt_sm = torch.bmm(J, grad_pi)
    
    self.grad_input = torch.bmm(grad_o_wrt_sm, W):permute(2,1,3)
    self.grad_input = self.grad_input + grad_M

    self.gradBias:copy(grad_o_wrt_sm:sum(1):sum(2):view(1))
    
    self.gradWeight:copy(torch.bmm(M:view(maxSteps,batchSize,d):permute(2,3,1), grad_o_wrt_sm):sum(1):view(d))

    local gradInputSorted = torch.Tensor():resizeAs(M):typeAs(M):zero()

    for i=1,maxSteps do
        for b=1,batchSize do
            gradInputSorted[self.indices[i][b]][b]:copy(self.grad_input[i][b])
        end
    end
    return gradInputSorted
end
