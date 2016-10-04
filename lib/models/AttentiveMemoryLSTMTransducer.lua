local AttentiveMemoryLSTMTransducer = 
    torch.class('nn.memory.AttentiveMemoryLSTMTransducer')

function AttentiveMemoryLSTMTransducer:__init(inputVocabSize, outputVocabSize, 
        dimSize, numLayers, optimState, mode)

    self.mode = mode or "all"
    assert(self.mode == "all" or self.mode == "forward" 
        or self.mode == "backward")

    print(self.mode)

    self.inputVocabSize = inputVocabSize
    self.outputVocabSize = outputVocabSize
    self.dimSize = dimSize
    self.numLayers = numLayers
    self.optimState = optimState
    self:buildNetwork()
end

function AttentiveMemoryLSTMTransducer:buildNetwork()
    local inputVocabSize = self.inputVocabSize
    local outputVocabSize = self.outputVocabSize
    local dimSize = self.dimSize
    local numLayers = self.numLayers
    local mode = self.mode

    self.encoderLookup = nn.LookupTableMaskZero(inputVocabSize, dimSize)
    self.decoderLookup = nn.LookupTableMaskZero(outputVocabSize, dimSize)

    self.coupledLSTM = nn.Sequential():add(
        nn.ParallelTable():add(self.encoderLookup):add(self.decoderLookup))
    self.coupledLSTM:add(nn.CoupledLSTM(dimSize, numLayers))

    self.memoryLinearLayer = 
        nn.Sequencer(
            nn.MaskZero(
                nn.Linear(dimSize, outputVocabSize, false), 1))
    self.decoderLinearLayer = 
        nn.Sequencer(
            nn.MaskZero(
                nn.Linear(dimSize, outputVocabSize), 1))

    self.softMaxLayer = nn.Sequencer(nn.MaskZero(nn.LogSoftMax(), 1))

    self.net = nn.Sequential()
    self.net:add(self.coupledLSTM)
    self.net:add(
        nn.ParallelTable():add(
            nn.LinearAssociativeMemoryWriter(dimSize, mode)):add(
            nn.Identity()))
    self.net:add(
        nn.ConcatTable():add(
            nn.Sequential():add(
                nn.LinearAssociativeMemoryReader(dimSize):maskZero()):add(
                self.memoryLinearLayer)):add(
            nn.Sequential():add(
                nn.SelectTable(2)):add(
                self.decoderLinearLayer)))
    self.net:add(nn.CAddTable())
    self.net:add(self.softMaxLayer)

    local params, gradParams = self.net:getParameters()
    self.params = params
    self.gradParams = gradParams
    
    self.coupledLSTM:get(2):couple()

    self.criterion = nn.SequencerCriterion(
        nn.MaskZeroCriterion(nn.ClassNLLCriterion(nil, false),1))

end

function AttentiveMemoryLSTMTransducer:cuda()
    self.net = self.net:cuda()
    self.criterion = self.criterion:cuda()
    local params, gradParams = self.net:getParameters()
    self.params = params
    self.gradParams = gradParams
    
    return self
end

function AttentiveMemoryLSTMTransducer:train(encIn, decIn, decOut)
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

function AttentiveMemoryLSTMTransducer:loss(encIn, decIn, decOut)
    local input = {encIn:t(), decIn:t()}
    local output = decOut:t()
    local outputPred = self.net:forward(input)
    local err = self.criterion:forward(outputPred, output)
    return err
end
