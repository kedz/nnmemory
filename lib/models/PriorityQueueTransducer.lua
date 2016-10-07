local PriorityQueueTransducer = 
    torch.class('nn.memory.PriorityQueueTransducer')

function PriorityQueueTransducer:__init(inputVocabSize, outputVocabSize, 
        dimSize, numQueues, numLayers, optimState, readBiasInit, 
        forgetBiasInit)
    self.inputVocabSize = inputVocabSize
    self.outputVocabSize = outputVocabSize
    self.dimSize = dimSize
    self.numLayers = numLayers
    self.numQueues = numQueues
    self.optimState = optimState
    self.readBiasInit = readBiasInit
    self.forgetBiasInit = forgetBiasInit

    self:buildNetwork()
end

function PriorityQueueTransducer:buildNetwork()
    local inputVocabSize = self.inputVocabSize
    local outputVocabSize = self.outputVocabSize
    local dimSize = self.dimSize
    local numLayers = self.numLayers
    local numQueues = self.numQueues
    local readBiasInit = self.readBiasInit
    local forgetBiasInit = self.forgetBiasInit

    self.encoderLookup = nn.LookupTableMaskZero(inputVocabSize, dimSize)
    self.decoderLookup = nn.LookupTableMaskZero(outputVocabSize, dimSize)

    self.coupledLSTM = nn.Sequential():add(
        nn.ParallelTable():add(self.encoderLookup):add(self.decoderLookup))
    self.coupledLSTM:add(nn.CoupledLSTM(dimSize, numLayers))


    self.cells = {}
    self.queueEncoders = {}
    self.queueDecoders = {}

    for i=1,numQueues do
        local memoryCell = nn.MemoryCell()
        memoryCell:add(nn.LinearAssociativeMemoryWriterP(dimSize, "all"))
        memoryCell:maskZero()
        self.cells[i] = memoryCell

        local priorityQueueEnc = nn.Sequential():add(
            nn.ParallelTable():add(
                nn.Sequential():add(memoryCell):add(nn.SortOnKey(true))):add(
                nn.Identity())):add(
            nn.FlattenTable()) 
        self.queueEncoders[i] = priorityQueueEnc

        self.queueDecoders[i] = 
            nn.PriorityQueueSimpleDecoder(dimSize, readBiasInit, 
                forgetBiasInit):maskZero()

    end
    
    self.priorityQueue = nn.Sequential():add(
        nn.ConcatTable()):add(
        nn.ParallelTable())
    
    for i=1,numQueues do
        self.priorityQueue:get(1):add(self.queueEncoders[i])
        self.priorityQueue:get(2):add(self.queueDecoders[i])
    end
    self.priorityQueue:add(nn.CAddTable())

--    self.priorityQueue:add(
--        nn.ConcatTable():add(
--            nn.PriorityQueueSimpleDecoder(dimSize):maskZero()):add(
--            nn.SelectTable(3)))

    self.priorityQueueLinearLayer = 
        nn.Sequencer(
            nn.MaskZero(
                nn.Linear(dimSize, outputVocabSize, false), 1))

    self.decoderLinearLayer = 
        nn.Sequencer(
            nn.MaskZero(
                nn.Linear(dimSize, outputVocabSize), 1))



    self.softMaxLayer = nn.Sequencer(nn.MaskZero(nn.LogSoftMax(), 1))


    self.coupledLSTMNet = nn.Sequential()
    self.coupledLSTMNet:add(self.coupledLSTM)
    self.coupledLSTMNet:add(nn.SelectTable(2))
    self.coupledLSTMNet:add(self.decoderLinearLayer)
    self.coupledLSTMNet:add(self.softMaxLayer)


    self.priorityQueueNet = nn.Sequential()
    self.priorityQueueNet:add(self.coupledLSTM)
    self.priorityQueueNet:add(
        nn.ConcatTable():add(
            self.priorityQueue):add(
            nn.SelectTable(2)))
    self.priorityQueueNet:add(
        nn.ParallelTable():add(
            self.priorityQueueLinearLayer):add(
            self.decoderLinearLayer))
    self.priorityQueueNet:add(nn.CAddTable())
    self.priorityQueueNet:add(self.softMaxLayer)

    local params, gradParams = self.priorityQueueNet:getParameters()
    self.params = params
    self.gradParams = gradParams

    self.criterion = nn.SequencerCriterion(
        nn.MaskZeroCriterion(nn.ClassNLLCriterion(nil, false),1))

end

function PriorityQueueTransducer:cuda()
    self.coupledLSTMNet = self.coupledLSTMNet:cuda()
    self.priorityQueueNet = self.priorityQueueNet:cuda()
    self.criterion = self.criterion:cuda()
    local params, gradParams = self.priorityQueueNet:getParameters()
    self.params = params
    self.gradParams = gradParams
    
    return self
end

function PriorityQueueTransducer:lossModel(model, encIn, decIn, decOut)
    local input = {encIn:t(), decIn:t()}
    local output = decOut:t()
    local outputPred = model:forward(input)
    local err = self.criterion:forward(outputPred, output)
    return err
end

function PriorityQueueTransducer:lossCoupledLSTM(encIn, decIn, decOut)
    self.coupledLSTM:get(2):couple()
    return self:lossModel(self.coupledLSTMNet, encIn, decIn, decOut)
end

function PriorityQueueTransducer:lossPriorityQueue(encIn, decIn, decOut)
    return self:lossModel(self.priorityQueueNet, encIn, decIn, decOut)
end

function PriorityQueueTransducer:trainModel(model, encIn, decIn, decOut)
    local input = {encIn:t(), decIn:t()}
    local output = decOut:t()

    local function feval(params)
        model:zeroGradParameters()
        local outputPred = model:forward(input)
        local err = self.criterion:forward(outputPred, output)
        local gradOutput = self.criterion:backward(outputPred, output)
        model:backward(input, gradOutput)
        return err, self.gradParams
    end
    local _, loss = optim.adam(feval, self.params, self.optimState)
    return loss[1]
end


function PriorityQueueTransducer:trainCoupledLSTM(encIn, decIn, decOut)
    self.coupledLSTM:get(2):couple()
    return self:trainModel(self.coupledLSTMNet, encIn, decIn, decOut)
end

function PriorityQueueTransducer:trainPriorityQueue(encIn, decIn, decOut)
    return self:trainModel(self.priorityQueueNet, encIn, decIn, decOut)
end




