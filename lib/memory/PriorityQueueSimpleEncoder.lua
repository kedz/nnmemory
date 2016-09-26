local PriorityQueueSimpleEncoder, Parent = 
    torch.class('nn.PriorityQueueSimpleEncoder', 'nn.Module')

function PriorityQueueSimpleEncoder:__init(dimSize)
    Parent.__init(self)
    self.dimSize = dimSize

    self.weight = torch.Tensor(dimSize)    
    self.gradWeight = torch.Tensor(dimSize)    
    self.bias = torch.Tensor(1)    
    self.gradBias = torch.Tensor(1)    

    self.pi = torch.Tensor()
    self.Zbuffer = torch.Tensor()
    self.Ibuffer = torch.Tensor()
    self.Jbuffer = torch.Tensor()
    self.indices = torch.LongTensor()
    self.inverse_indices = torch.LongTensor()
    self.pi_sorted = torch.Tensor()
    self.M_sorted = torch.Tensor()

    self.grad_o_wrt_sm_buf = torch.Tensor()
    self.grad_input_unsorted = torch.Tensor()
    self.grad_buffer1 = torch.Tensor()
    self.gradInput = torch.Tensor()

    self.isMaskZero = false
    self.isCuda = false
    
    self:reset()

end

function PriorityQueueSimpleEncoder:cuda()
    Parent.cuda(self)
    self.indices = torch.CudaLongTensor()
    self.inverse_indices = torch.CudaLongTensor()
    self.isCuda = true
    return self
end

function PriorityQueueSimpleEncoder:float()
    Parent.float(self)
    self.indices = torch.LongTensor()
    self.inverse_indices = torch.LongTensor()
    self.isCuda = false
    return self
end

function PriorityQueueSimpleEncoder:double()
    Parent.double(self)
    self.indices = torch.LongTensor()
    self.inverse_indices = torch.LongTensor()
    self.isCuda = false
    return self
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
    local isCuda = false
    if input:type() == "torch.CudaTensor" then
        isCuda = true
        if self.indices:type() ~= "torch.CudaLongTensor" then
            self.indices = self.indices:type("torch.CudaLongTensor")
            self.inverse_indices = self.inverse_indices:type(
                "torch.CudaLongTensor")
        end
    end
    local d = self.dimSize
    local maxSteps = input:size(1)
    local batchSize = input:size(2)

    local W = self.weight:view(1,d,1):expand(batchSize, d, 1)
    local b = self.bias:view(1,1,1):expand(batchSize, 1, 1)
    local pi = self.pi:resize(maxSteps, batchSize, 1):zero()

    local X = input:view(maxSteps, batchSize, 1, d)

    for t=1,maxSteps do

        local pi_t = pi[t]:view(batchSize, 1, 1) 
        torch.baddbmm(pi_t, b, X[t], W)
    end

    pi = pi:view(maxSteps,batchSize)

    if self.isMaskZero then
        local mask
        if  isCuda then
            mask = torch.eq(input[{{},{},{1}}], 0):long():nonzero()
        else
            mask = torch.eq(input[{{},{},{1}}], 0):nonzero()
        end
        if mask:dim() > 0 then 
            mask = mask[{{},{1,2}}]
            for m=1,mask:size(1) do
                pi[mask[m][1]][mask[m][2]] = -math.huge
            end
        end
    end

    -- softmax over time
    torch.exp(pi, pi)
    torch.cdiv(
        pi, 
        pi, 
        torch.sum(self.Zbuffer, pi, 1):expand(maxSteps,batchSize))
    
    -- sort priorities
    local pi_sorted, indices = torch.sort(
        self.pi_sorted, 
        self.indices, 
        pi, 1, true)
        
    -- sort memory (base on priority)
    local memory_sorted = self.M_sorted:resizeAs(input):typeAs(input):zero()

    for b=1,batchSize do
        memory_sorted:select(2,b):index(
            input:select(2,b), 1, indices:select(2,b))
    end

    self.output = {memory_sorted, pi_sorted}
    return self.output
end

function PriorityQueueSimpleEncoder:updateGradInput(input, gradOutput)

    local isCuda = false
    if input:type() == "torch.CudaTensor" then
        isCuda = true
    end
    local d = self.dimSize

    local grad_M = gradOutput[1]
    local M = self.output[1]
    local pi = self.output[2]

    local maxSteps = input:size(1)
    local batchSize = input:size(2)
    local W = self.weight:view(1,1,d):expand(batchSize,1,d)

    local I 
    if isCuda then
        if self.Ibuffer:nElement() < maxSteps * maxSteps then
            self.Ibuffer = self.Ibuffer:resize(maxSteps, maxSteps,1):zero()
            for i=1,maxSteps do
                self.Ibuffer[i][i][1] = 1
            end
        end 

        I = self.Ibuffer[{{1,maxSteps}, {1,maxSteps},{}}]:expand(
                maxSteps, maxSteps, batchSize)
    else
        I = torch.eye(self.Ibuffer, maxSteps):view(
            maxSteps, maxSteps, 1):expand(maxSteps, maxSteps, batchSize)
    end

    -- compute jacobian of softmax
    local J = torch.csub(
        self.Jbuffer, I, 
        pi:view(1,maxSteps,batchSize):expand(maxSteps,maxSteps,batchSize))
    J = torch.cmul(
        J, J,
        pi:view(maxSteps,1,batchSize):expand(maxSteps,maxSteps,batchSize))
    J = J:permute(3,1,2)

    local grad_pi = gradOutput[2]:permute(2,1):contiguous()

    --local grad_pi = grad_pi:view(maxSteps, batchSize, 1):permute(2,1,3)
    local grad_pi = grad_pi:view(batchSize,maxSteps,1)

    -- For some reason batch matrix multiply won't resize storage automatically?
    self.grad_o_wrt_sm_buf = self.grad_o_wrt_sm_buf:resize(batchSize, maxSteps, 1)
    local grad_o_wrt_sm = torch.bmm(
        self.grad_o_wrt_sm_buf,
        J, 
        grad_pi)
    
    -- For some reason batch matrix multiply won't resize storage automatically?
    self.grad_input_unsorted = 
        self.grad_input_unsorted:resize(batchSize, maxSteps, d)
    self.grad_input_unsorted = torch.bmm(
        self.grad_input_unsorted, 
        grad_o_wrt_sm, 
        W):permute(2,1,3)
    self.grad_input_unsorted:add(grad_M)

    torch.sum(self.grad_buffer1, grad_o_wrt_sm, 1)
    torch.sum(self.gradBias:view(1,1,1), self.grad_buffer1, 2)
    
    -- For some reason batch matrix multiply won't resize storage automatically?
    self.grad_buffer1 = self.grad_buffer1:resize(batchSize, d, 1)
    torch.bmm(
        self.grad_buffer1,
        M:permute(2,3,1), 
        grad_o_wrt_sm)
    torch.sum(self.gradWeight:view(1,d,1), self.grad_buffer1, 1)    

    local gradInput = self.gradInput:resizeAs(M):zero()

    local inverse_indices = self.inverse_indices:resizeAs(self.indices)
    for i=1,maxSteps do
        for b=1,batchSize do
            inverse_indices[self.indices[i][b]][b] = i
        end
    end

    for b=1,batchSize do
        gradInput:select(2,b):index(self.grad_input_unsorted:select(2,b), 1, inverse_indices:select(2,b))
    end
    return gradInput
end
