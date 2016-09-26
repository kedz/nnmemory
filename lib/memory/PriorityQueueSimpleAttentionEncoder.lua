local PriorityQueueSimpleAttentionEncoder, Parent = 
    torch.class('nn.PriorityQueueSimpleAttentionEncoder', 'nn.Module')

function PriorityQueueSimpleAttentionEncoder:__init(dimSize)
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

    self.buffer1 = torch.Tensor()
    self.grad_pi_inv = torch.Tensor()
    self.grad_M_inv = torch.Tensor()



    self.isMaskZero = false
    self.isCuda = false
    
    self:reset()

end

function PriorityQueueSimpleAttentionEncoder:cuda()
    Parent.cuda(self)
    self.indices = torch.CudaLongTensor()
    self.inverse_indices = torch.CudaLongTensor()
    self.isCuda = true
    return self
end

function PriorityQueueSimpleAttentionEncoder:float()
    Parent.float(self)
    self.indices = torch.LongTensor()
    self.inverse_indices = torch.LongTensor()
    self.isCuda = false
    return self
end

function PriorityQueueSimpleAttentionEncoder:double()
    Parent.double(self)
    self.indices = torch.LongTensor()
    self.inverse_indices = torch.LongTensor()
    self.isCuda = false
    return self
end

function PriorityQueueSimpleAttentionEncoder:maskZero()
    self.isMaskZero = true
    return self
end

function PriorityQueueSimpleAttentionEncoder:reset()
    self.weight:uniform(-1, 1)
    self.bias:fill(1.0) -- initial bias is to write
    self:zeroGradParameters()
end

function PriorityQueueSimpleAttentionEncoder:zeroGradParameters()
    self.gradWeight:zero()
    self.gradBias:zero()
end

function PriorityQueueSimpleAttentionEncoder:updateOutput(input)
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
    local Xcum = self.buffer1:resizeAs(X[1]):zero()

    torch.baddbmm(pi[maxSteps]:view(batchSize, 1, 1), b, X[maxSteps], W)
    for t=maxSteps-1,1,-1 do
        local pi_t = pi[t]:view(batchSize, 1, 1) 
        torch.baddbmm(pi_t, b, X[t], W)
        Xcum:add(X[t+1])
        pi_t:add(torch.bmm(X[t], Xcum:permute(1,3,2)))
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
    self.memory_orig = input
    self.pi_orig = pi
    self.output = {memory_sorted, pi_sorted}
    return self.output
end

function PriorityQueueSimpleAttentionEncoder:updateGradInput(input, gradOutput)

    local isCuda = false
    if input:type() == "torch.CudaTensor" then
        isCuda = true
    end
    local d = self.dimSize

    local grad_M_sorted = gradOutput[1]
    local grad_pi_sorted = gradOutput[2]

    --    local M = self.output[1]
--    local pi = self.output[2]

    local maxSteps = input:size(1)
    local batchSize = input:size(2)
    local X = input:view(maxSteps, batchSize, 1, d)
    local W = self.weight:view(1,1,d):expand(batchSize,1,d)

    local inverse_indices = self.inverse_indices:resizeAs(self.indices)
    for i=1,maxSteps do
        for b=1,batchSize do
            inverse_indices[self.indices[i][b]][b] = i
        end
    end
    local grad_M_inv = self.grad_M_inv:resizeAs(grad_M_sorted):zero()
    local grad_pi_inv = self.grad_pi_inv:resizeAs(grad_pi_sorted):zero()

    for b=1,batchSize do
        grad_M_inv:select(2,b):index(grad_M_sorted:select(2,b), 1, inverse_indices:select(2,b))
        grad_pi_inv:select(2,b):index(grad_pi_sorted:select(2,b), 1, inverse_indices:select(2,b))
    end
    local grad_M = grad_M_inv
    local grad_pi = grad_pi_inv
    local pi = self.pi_orig
    local M = self.memory_orig


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

    local grad_pi = grad_pi:permute(2,1):contiguous()

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
    self.grad_input_unsorted = torch.cmul(grad_o_wrt_sm:expand(batchSize, maxSteps, d),
        W:expand(batchSize, maxSteps, d))
    self.grad_input_unsorted = self.grad_input_unsorted:contiguous()
    self.grad_input_unsorted = self.grad_input_unsorted + grad_M:permute(2,1,3)

    --torch.sum(self.grad_buffer1, grad_o_wrt_sm, 1)
    --torch.sum(self.gradBias:view(1,1,1), self.grad_buffer1, 2)
    self.gradBias:copy(grad_o_wrt_sm:sum(1):sum(2)) 
    -- For some reason batch matrix multiply won't resize storage automatically?
    self.gradWeight:copy(torch.cmul(grad_o_wrt_sm:expand(batchSize, maxSteps, d),
        M:permute(2,1,3)):sum(1):sum(2))

    local Xcum = self.buffer1:zero()
    local grad_sm = grad_o_wrt_sm:view(batchSize, maxSteps, 1):expand(batchSize, maxSteps, d):permute(2,1,3)
    self.grad_input_unsorted = self.grad_input_unsorted:permute(2,1,3):contiguous()
    
    for t=maxSteps-1,1,-1 do
        Xcum:add(X[t+1])
        self.grad_input_unsorted[t]:add(
            torch.cmul(grad_sm[t], Xcum:view(batchSize,d))) -- ):view(batchSize, d))
        self.grad_input_unsorted[{{t+1,maxSteps},{},{}}]:add(
            torch.cmul(grad_sm[t], X[t]:view(batchSize, d)):view(1,batchSize,d):expand(maxSteps - t, batchSize, d))
    end
    return self.grad_input_unsorted
end
