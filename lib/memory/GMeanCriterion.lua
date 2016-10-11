local GMeanCriterion, parent = torch.class('nn.GMeanCriterion', 'nn.Criterion')

function GMeanCriterion:__init(weight)
    parent.__init(self)
    self.weight = weight
    self.gradOutput = torch.Tensor(1):fill(1)

    self.mod = nn.Sequential()
    self.mod:add(nn.Log())
    self.mod:add(nn.Mean(2))
    self.mod:add(nn.Exp())
    self.mod:add(nn.Sum())
    self.mod:add(nn.MulConstant(weight))
end

function GMeanCriterion:updateOutput(input, target)
    self.output = self.mod:forward(input)[1]
    return self.output
end

function GMeanCriterion:updateGradInput(input, target)
    self.gradInput = self.mod:backward(input, self.gradOutput)
    return self.gradInput
end
