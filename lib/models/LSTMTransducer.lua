local LSTMTransducer = torch.class('nn.memory.LSTMTransducer')

function LSTMTransducer:__init(inputVocabSize, outputVocabSize, dimSize,
        numLayers, optimState)
    self.inputVocabSize = inputVocabSize
    self.outputVocabSize = outputVocabSize
    self.dimSize = dimSize
    self.numLayers = numLayers
    self.optimState = optimState
    self:buildNetwork()
end

function LSTMTransducer:buildNetwork()
    local inputVocabSize = self.inputVocabSize
    local outputVocabSize = self.outputVocabSize
    local dimSize = self.dimSize
    local numLayers = self.numLayers

    self.encoderLookup = nn.LookupTableMaskZero(inputVocabSize, dimSize)
    self.decoderLookup = nn.LookupTableMaskZero(outputVocabSize, dimSize)

    self.coupledLSTM = nn.Sequential():add(
        nn.ParallelTable():add(self.encoderLookup):add(self.decoderLookup))
    self.coupledLSTM:add(nn.CoupledLSTM(dimSize, numLayers))

    self.decoderLinearLayer = 
        nn.Sequencer(
            nn.MaskZero(
                nn.Linear(dimSize, outputVocabSize), 1))

    self.softMaxLayer = nn.Sequencer(nn.MaskZero(nn.LogSoftMax(), 1))

    self.net = nn.Sequential()
    self.net:add(self.coupledLSTM)
    self.net:add(nn.SelectTable(2))
    self.net:add(self.decoderLinearLayer)
    self.net:add(self.softMaxLayer)

    local params, gradParams = self.net:getParameters()
    self.params = params
    self.gradParams = gradParams
    
    self.coupledLSTM:get(2):couple()

    self.criterion = nn.SequencerCriterion(
        nn.MaskZeroCriterion(nn.ClassNLLCriterion(nil, false),1))

end

function LSTMTransducer:cuda()
    self.net = self.net:cuda()
    self.criterion = self.criterion:cuda()
    return self
end

function LSTMTransducer:train(encIn, decIn, decOut)
    local input = {encIn:t(), decIn:t()}
    local output = decOut:t()

    local function feval(params)
        self.net:zeroGradParameters()
        local outputPred = self.net:forward(input)
        local err = self.criterion:forward(outputPred, output)
        local gradOutput = self.criterion:backward(outputPred, output)
        self.net:backward(input, gradOutput)
        return err, self.gradParams
    end
    local _, loss = optim.adam(feval, self.params, self.optimState)
    return loss[1]
end

function LSTMTransducer:loss(encIn, decIn, decOut)
    local input = {encIn:t(), decIn:t()}
    local output = decOut:t()
    local outputPred = self.net:forward(input)
    local err = self.criterion:forward(outputPred, output)
    return err
end
