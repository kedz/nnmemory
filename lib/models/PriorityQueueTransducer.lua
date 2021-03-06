local PriorityQueueTransducer = 
    torch.class('nn.memory.PriorityQueueTransducer')

function PriorityQueueTransducer:__init(inputVocabSize, outputVocabSize, 
        dimSize, numQueues, numLayers, optimState, readBiasInit, 
        forgetBiasInit, interQueuePenaltyWeight, intraQueuePenaltyWeight)
    self.inputVocabSize = inputVocabSize
    self.outputVocabSize = outputVocabSize
    self.dimSize = dimSize
    self.numLayers = numLayers
    self.numQueues = numQueues
    self.optimState = optimState
    self.readBiasInit = readBiasInit
    self.forgetBiasInit = forgetBiasInit
    self.interQueuePenaltyWeight = interQueuePenaltyWeight
    self.intraQueuePenaltyWeight = intraQueuePenaltyWeight

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
    local interQueuePenaltyWeight = self.interQueuePenaltyWeight
    local intraQueuePenaltyWeight = self.intraQueuePenaltyWeight

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
                nn.Sequential():add(memoryCell):add(
                    nn.ConcatTable():add(
                        nn.SelectTable(2)):add( 
                        nn.SortOnKey(true)))):add(
                nn.Identity())):add(
            nn.FlattenTable()) 
        self.queueEncoders[i] = priorityQueueEnc

        self.queueDecoders[i] = 
            nn.PriorityQueueSimpleDecoder(dimSize, readBiasInit, 
                forgetBiasInit):maskZero()

    end
    
    self.priorityQueue = nn.Sequential():add(
        nn.ConcatTable())
        
    local decoders = nn.ParallelTable()
    for i=1,numQueues do
        self.priorityQueue:get(1):add(self.queueEncoders[i])
        decoders:add(
            nn.Sequential():add(
                nn.NarrowTable(2,3)):add(
                self.queueDecoders[i]))
    end
    local priorities = nn.ParallelTable()
    for q=1,self.numQueues do
        priorities:add(nn.SelectTable(1))
    end
    decoders = nn.Sequential():add(decoders):add(nn.CAddTable())
    self.priorityQueue:add(
        nn.ConcatTable():add(priorities):add(decoders))

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
        nn.ConcatTable():add(
            nn.Sequential():add(nn.SelectTable(1)):add(nn.SelectTable(1))):add(
            nn.Sequential():add(nn.SelectTable(1)):add(nn.SelectTable(2))):add(
            nn.Sequential():add(nn.SelectTable(2))))

    self.priorityQueueNet:add(
        nn.ParallelTable():add(
            nn.Identity()):add(
            self.priorityQueueLinearLayer):add(
            self.decoderLinearLayer))
    self.priorityQueueNet:add(
        nn.ConcatTable():add(
            nn.SelectTable(1)):add(
        nn.Sequential():add(
            nn.ConcatTable():add(
                nn.SelectTable(2)):add(
                nn.SelectTable(3))):add(
            nn.CAddTable()):add(
            self.softMaxLayer)))

    self.priorityQueueNet:add(
        nn.ConcatTable():add(
            nn.Sequential():add(
                nn.SelectTable(1)):add(
                nn.MapTable():add(nn.Unsqueeze(1))):add(
                nn.JoinTable(1)):add(
                nn.Transpose({3,2}))):add(
            nn.SelectTable(2)))            

    self.priorityQueueNet:add(
        nn.ConcatTable():add(
            nn.Sequential():add(
                nn.SelectTable(1)):add(
                nn.Transpose({3,1}))):add(
            nn.SelectTable(1)):add(
            nn.SelectTable(2)))

         

    local params, gradParams = self.priorityQueueNet:getParameters()
    self.params = params
    self.gradParams = gradParams

    self.nllPenalty = nn.SequencerCriterion(
        nn.MaskZeroCriterion(nn.ClassNLLCriterion(nil, false),1))
    self.interQueuePenalty = nn.SequencerCriterion(nn.GMeanCriterion(1.0))
    self.intraQueuePenalty = nn.SequencerCriterion(nn.GMeanCriterion(1.0))
    self.criterion = nn.ParallelCriterion():add(
        self.interQueuePenalty, interQueuePenaltyWeight):add(
        self.intraQueuePenalty, intraQueuePenaltyWeight):add(
        self.nllPenalty, 1.0)

    self.interQueuePenaltyTarget = torch.Tensor()
    self.intraQueuePenaltyTarget = torch.Tensor()

end

function PriorityQueueTransducer:cuda()
    self.coupledLSTMNet = self.coupledLSTMNet:cuda()
    self.priorityQueueNet = self.priorityQueueNet:cuda()
    self.criterion = self.criterion:cuda()
    local params, gradParams = self.priorityQueueNet:getParameters()
    self.params = params
    self.gradParams = gradParams
    self.interQueuePenaltyTarget = self.interQueuePenaltyTarget:cuda()
    self.intraQueuePenaltyTarget = self.intraQueuePenaltyTarget:cuda()
    
    return self
end

function PriorityQueueTransducer:lossModel(model, encIn, decIn, decOut)
    local input = {encIn:t(), decIn:t()}
    local output = {
        self.interQueuePenaltyTarget:resize(encIn:size(2), encIn:size(1)),
        self.intraQueuePenaltyTarget:resize(self.numQueues, encIn:size(1)),
        decOut:t()}
    local outputPred = model:forward(input)
    local err = self.criterion:forward(outputPred, output)
    return self.criterion.criterions[3].output
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
    local output = {
        self.interQueuePenaltyTarget:resize(encIn:size(2), encIn:size(1)),
        self.intraQueuePenaltyTarget:resize(self.numQueues, encIn:size(1)),
        decOut:t()}

    local function feval(params)
        model:zeroGradParameters()
        local outputPred = model:forward(input)
        local err = self.criterion:forward(outputPred, output)
        local gradOutput = self.criterion:backward(outputPred, output)
        model:backward(input, gradOutput)
        return err, self.gradParams
    end
    local _, loss = optim.adam(feval, self.params, self.optimState)
    return self.criterion.criterions[3].output
end


function PriorityQueueTransducer:trainCoupledLSTM(encIn, decIn, decOut)
    self.coupledLSTM:get(2):couple()
    return self:trainModel(self.coupledLSTMNet, encIn, decIn, decOut)
end

function PriorityQueueTransducer:trainPriorityQueue(encIn, decIn, decOut)
    return self:trainModel(self.priorityQueueNet, encIn, decIn, decOut)
end


function PriorityQueueTransducer:info(encIn, decIn, decOut)
    local input = {encIn:t(), decIn:t()}
    local output = decOut:t()
    self.priorityQueueNet:clearState() 
    local outputPred = self.priorityQueueNet:forward(input) 
    local indicesInverted = {}
    for q=1,self.numQueues do
        local indS = self.queueEncoders[q]:get(1):get(1):get(2):get(2).indices_sorted
        local _, indInv = torch.sort(indS, 1)
        indicesInverted[q] = indS
    end
    myInfo = {}
    for b=1,encIn:size(1) do
        myInfo[b] = {}
        for q=1,self.numQueues do
            myInfo[b][q] = {elements={}}
            
            myInfo[b][q]["read"] = {} 
            for t=1,self.queueDecoders[q].R:size(1) do
                if self.queueDecoders[q].R[t][b] == 0 then break end
                myInfo[b][q].read[t] = self.queueDecoders[q].R[t][b]
            end
            myInfo[b][q]["forget"] = {}
            for t=1,self.queueDecoders[q].F:size(1) do
                if self.queueDecoders[q].F[t][b] == 0 then break end
                myInfo[b][q].forget[t] = self.queueDecoders[q].F[t][b]
            end
            inputIds = encIn[b]:index(1, indicesInverted[q]:t()[b])
            for i=1,inputIds:size(1) do
                if inputIds[i] == 0 then break end
                local att = {}
                for t=1,#myInfo[b][q].read do
                    att[t] = self.queueDecoders[q].P[t][i][b]
                end
                myInfo[b][q]["elements"][i] = {
                    id=inputIds[i],
                    attention=att}
                
            
            --for t=1,1 do
                --print(self.queueDecoders[q].P[t]:t()[b])
            end
        end
    end
    return myInfo
end


