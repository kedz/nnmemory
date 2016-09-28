local CoupledLSTM, Parent = torch.class('nn.CoupledLSTM', 'nn.Module')

function CoupledLSTM:__init(dimSize, numLayers)

    self.dimSize = dimSize
    self.numLayers = numLayers
    self.isCoupled = true

    self.encoder = nn.Sequential()
    for i=1,numLayers do
        local m = nn.SeqLSTM(dimSize, dimSize)
        m:maskZero()
        self.encoder:add(m)
    
    end
    self.decoder = nn.Sequential()
    for i=1,numLayers do
        local m = nn.SeqLSTM(dimSize, dimSize)
        m:maskZero()
        self.decoder:add(m)
    end
    
end

function CoupledLSTM:parameters()
    local params, gradParams = self.encoder:parameters()
    local decParams, decGradParams = self.decoder:parameters()
    for i=1,#decParams do
        table.insert(params, decParams[i])
        table.insert(gradParams, decGradParams[i])
    end
    return params, gradParams
end

function CoupledLSTM:couple()
    self.isCoupled = true
    return self
end

function CoupledLSTM:decouple()
    self.isCoupled = false
    return self
end

function CoupledLSTM:forwardConnect()
    for i=1,self.numLayers do
        self.decoder:get(i).userPrevOutput = 
            self.encoder:get(i).output[-1]
        self.decoder:get(i).userPrevCell = 
            self.encoder:get(i).cell[-1]
   end
end

function CoupledLSTM:backwardConnect()
    for i=1,self.numLayers do
        self.encoder:get(i).userNextGradCell = 
            self.decoder:get(i).userGradPrevCell
        self.encoder:get(i).gradPrevOutput = 
            self.decoder:get(i).userGradPrevOutput
   end
end

function CoupledLSTM:updateOutput(input)

    self.encoderOutput = self.encoder:forward(input[1])
    if self.isCoupled then self:forwardConnect() end
    self.decoderOutput = self.decoder:forward(input[2])
    
    self.output = {self.encoderOutput, self.decoderOutput}
    return self.output
end

function CoupledLSTM:updateGradInput(input, gradOutput)
    local gradDecoder = self.decoder:backward(input[2], gradOutput[2])
    if self.isCoupled then self:backwardConnect() end
    local gradEncoder = self.encoder:backward(input[1], gradOutput[1])
    self.gradInput = {gradEncoder, gradDecoder}
    return self.gradInput
end

--function CoupledLSTM:accGradParameters(input, gradOutput)
   
--end

function CoupledLSTM:zeroGradParameters()
    self.encoder:zeroGradParameters()
    self.decoder:zeroGradParameters()
end

