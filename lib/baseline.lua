local CoupledLSTM = torch.class('nn.CoupledLSTM')

function CoupledLSTM:__init(inputVocabSize, outputVocabSize, dimSize,
        optimState)
    self.inputVocabSize = inputVocabSize
    self.outputVocabSize = outputVocabSize
    self.dimSize = dimSize
    self.encoderLayers = 2
    self.decoderLayers = 2
    self.optimState = optimState
    self:buildNetwork()
    self:flatten()     

end

function CoupledLSTM:flatten()
    local params, gradParams = self.encoder:parameters()
    local decoderParams, decoderGradParams = self.decoder:parameters()
    for i=1,#decoderParams do
        table.insert(params, decoderParams[i])
        table.insert(gradParams, decoderGradParams[i])
    end
    self.params = self.encoder.flatten(params)
    self.gradParams = self.encoder.flatten(gradParams)
end

function CoupledLSTM:cuda()
    self.encoder:cuda()
    self.decoder:cuda()
    self.criterion:cuda()
    self:flatten()
end

function CoupledLSTM:buildNetwork()

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

    self.decoder = nn.Sequential()
    self.decoder:add(
        nn.LookupTableMaskZero(outputVocabSize, dimSize))

    self.decoder.lstmLayers = {}
    for i=1,decoderLayers do
        self.decoder.lstmLayers[i] = nn.SeqLSTM(dimSize, dimSize)
        self.decoder.lstmLayers[i]:maskZero()
        self.decoder:add(self.decoder.lstmLayers[i])
    end

    self.decoder:add(
        nn.Sequencer(
            nn.MaskZero(
                nn.Linear(dimSize, outputVocabSize), 1)))
        
    self.decoder:add(
        nn.Sequencer(
            nn.MaskZero(
                nn.LogSoftMax(), 1)))

    self.criterion = nn.SequencerCriterion(
        nn.MaskZeroCriterion(nn.ClassNLLCriterion(nil, false),1))

end

function CoupledLSTM:forwardConnect()
    for i=1,#self.encoder.lstmLayers do
        self.decoder.lstmLayers[i].userPrevOutput = 
            self.encoder.lstmLayers[i].output[-1]
        self.decoder.lstmLayers[i].userPrevCell = 
            self.encoder.lstmLayers[i].cell[-1]
   end
end

--[[ Backward coupling: Copy decoder gradients to encoder LSTM ]]--
function CoupledLSTM:backwardConnect()
    for i=1,#self.encoder.lstmLayers do
        self.encoder.lstmLayers[i].userNextGradCell = 
            self.decoder.lstmLayers[i].userGradPrevCell
        self.encoder.lstmLayers[i].gradPrevOutput = 
            self.decoder.lstmLayers[i].userGradPrevOutput
   end
end


function CoupledLSTM:loss(encoderIn, decoderIn, decoderOut)
    local encoderOut = self.encoder:forward(encoderIn:t())
    self:forwardConnect()
    local decoderOutPred = self.decoder:forward(decoderIn:t())
    local err = self.criterion:forward(decoderOutPred, decoderOut:t())
    return err
end

function CoupledLSTM:train(encoderIn, decoderIn, decoderOut)
    local function feval(params)
        self.gradParams:zero()
        local encoderOut = self.encoder:forward(encoderIn:t())
        self:forwardConnect()
        local decoderOutPred = self.decoder:forward(decoderIn:t())
        local err = self.criterion:forward(decoderOutPred, decoderOut:t())
        local gradOutput = 
            self.criterion:backward(decoderOutPred, decoderOut:t())
        self.decoder:backward(decoderIn:t(), gradOutput)
        self:backwardConnect()
        self.encoder:backward(encoderIn:t(), encoderOut:clone():zero())

        return err, self.gradParams
    end
    local _, loss = optim.adam(feval, self.params, self.optimState)
    return loss[1]
end
