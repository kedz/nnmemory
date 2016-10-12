local GMeanCriterion, parent = torch.class('nn.GMeanCriterion', 'nn.Criterion')

function GMeanCriterion:__init(weight)
    parent.__init(self)
    self.weight = weight
    self.gradOutput = torch.Tensor(1):fill(1)


    self.mod = nn.Sequential():add(
        nn.ParallelTable():add(
            nn.Sequential():add(
                nn.Log()):add(
                nn.Sum(2))):add(
            nn.Identity())):add(
        nn.CMulTable())
    self.mod:add(nn.Unsqueeze(2))  
    self.mod:add(nn.MaskZero(nn.Exp(), 1))
    self.mod:add(nn.Sum())
    self.mod:add(nn.MulConstant(weight))
end

function GMeanCriterion:updateOutput(input, target)
    local mask = torch.eq(input, 0)
    input[mask] = 1.0
    local denom = (input:size(2) - mask:typeAs(input):sum(2)):squeeze():pow(-1)
    denom[torch.eq(denom, math.huge)] = 1
    self.output = self.mod:forward({input, denom})[1]
    input[mask] = 0.0
  
    self.mask = mask 
    self.denom = denom 
    return self.output
end

function GMeanCriterion:updateGradInput(input, target)
    self.gradInput = self.mod:backward({input, self.denom}, self.gradOutput)[1]
    self.gradInput[self.mask] = 0.0
    return self.gradInput
end
