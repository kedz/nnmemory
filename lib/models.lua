
local PriorityQueueNetwork = torch.class('nn.PriorityQueueNetwork')

function PriorityQueueNetwork:__init(inputVocabSize, outputVocabSize, dimSize,
        optimState)
    self.inputVocabSize = inputVocabSize
    self.outputVocabSize = outputVocabSize
    self.dimSize = dimSize
    self.encoderLayers = 1
    self.decoderLayers = 1
    self.optimState = optimState
    self:buildNetwork()

    self.params, self.gradParams = self.net:getParameters()

end

function PriorityQueueNetwork:buildNetwork()

    local inputVocabSize = self.inputVocabSize
    local outputVocabSize = self.outputVocabSize
    local dimSize = self.dimSize
    local encoderLayers = self.encoderLayers
    local decoderLayers = self.decoderLayers

    self.encoder = nn.Sequential()
    self.encoder:add(
        nn.LookupTableMaskZero(inputVocabSize, dimSize))

    self.encoder.lstmLayers = {}
    for i=1,encoderLayers do
        self.encoder.lstmLayers[i] = nn.SeqLSTM(dimSize, dimSize)
        self.encoder.lstmLayers[i]:maskZero()
        self.encoder:add(self.encoder.lstmLayers[i])
    end
    self.encoder:add(nn.PriorityQueueSimpleEncoder(dimSize):maskZero())

    self.decoder = nn.Sequential()
    self.decoder:add(
        nn.LookupTableMaskZero(outputVocabSize, dimSize))

    self.decoder.lstmLayers = {}
    for i=1,decoderLayers do
        self.decoder.lstmLayers[i] = nn.SeqLSTM(dimSize, dimSize)
        self.decoder.lstmLayers[i]:maskZero()
        self.decoder:add(self.decoder.lstmLayers[i])
    end

    self.net = nn.Sequential()
    self.net:add(nn.ParallelTable():add(self.encoder):add(self.decoder))
    self.net:add(nn.FlattenTable())

    local join = nn.ConcatTable():add(
        nn.PriorityQueueSimpleDecoder(dimSize):maskZero())
    join:add(nn.SelectTable(3))
    self.net:add(join)
    self.net:add(nn.JoinTable(3))

    self.net:add(
        nn.Sequencer(
            nn.MaskZero(
                nn.Linear(2*dimSize, outputVocabSize), 1)))
        
    self.net:add(
        nn.Sequencer(
            nn.MaskZero(
                nn.LogSoftMax(), 1)))

    self.criterion = nn.SequencerCriterion(
        nn.MaskZeroCriterion(nn.ClassNLLCriterion(nil, false),1))

end

function PriorityQueueNetwork:encode(X)
    return self.encoder:forward(X:t())
end

function PriorityQueueNetwork:forward(input, output)
    local outputPred = self.net:forward(input)
    local err = self.criterion:forward(outputPred, output)
    local gradOutput = self.criterion:backward(outputPred, output)
    self.net:backward(input, gradOutput)
    print(err)
end

function PriorityQueueNetwork:loss(encoderIn, decoderIn, decoderOut)
    local input = {encoderIn:t(), decoderIn:t()}
    local decoderOutPred = self.net:forward(input)
    local err = self.criterion:forward(decoderOutPred, decoderOut:t())
    return err 
end

function PriorityQueueNetwork:train(encoderIn, decoderIn, decoderOut)
    local function feval(params)
        self.net:zeroGradParameters()
        
        local input = {encoderIn:t(), decoderIn:t()}
        local decoderOutPred = self.net:forward(input)
        local err = self.criterion:forward(decoderOutPred, decoderOut:t())
        local gradOutput = 
            self.criterion:backward(decoderOutPred, decoderOut:t())
        self.net:backward(input, gradOutput)
        return err, self.gradParams
    end
    local _, loss = optim.adam(feval, self.params, self.optimState)
    return loss[1]
end
