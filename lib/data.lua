data = {}

function data.readVocab(path, includeSpecial)
    -- vocab tables 
    local id2token = {}
    local token2id = {}
    local id = 1    

    if includeSpecial then
        id2token[1] ="<sos>"
        id2token[2] = "<eos>"
        token2id["<sos>"] = 1
        token2id["<eos>"] = 2
        id = 3
    end

    for token in io.lines(path) do
        if not token2id[token] then
            id2token[id] = token
            token2id[token] = id
            id = id + 1
        end
    end   
    return {id2token=id2token, token2id=token2id, size=#id2token}

end

function data.readData(path, inputVocab, outputVocab, readTopics)
    
    local maxInputSize = 0
    local maxOutputSize = 0
    local numExamples = 0
    local totalOutputSize = 0

    local inputs = {}
    local outputs = {}
    local outputSizes = {}
    for line in io.lines(path) do
        
        local input, outTokens, outTopics = unpack(stringx.split(line, " || "))

        input = stringx.split(input, " ")
        local output
        if readTopics then
            output = stringx.split(outTopics, " ")
        else
            output = stringx.split(outTokens, " ")
        end
        if #input > maxInputSize then maxInputSize = #input end
        if #output > maxOutputSize then maxOutputSize = #output end
        
        table.insert(inputs, input)
        table.insert(outputs, output)
        table.insert(outputSizes, #output)
        totalOutputSize = totalOutputSize + #output
    end

    numExamples = #inputs

    local encoderInput = torch.LongTensor(#inputs, maxInputSize):zero()
    local decoderInput = torch.LongTensor(#inputs, maxOutputSize + 1):zero()
    local decoderOutput = torch.LongTensor():resizeAs(decoderInput):zero()
    outputSizes = torch.LongTensor(outputSizes)
    
    for i=1,#inputs do
        local input = inputs[i]
        local output = outputs[i]
        local offset = maxInputSize - #input
        
        decoderInput[i][1] = 1
        decoderOutput[i][1 + #output] = 2
        for j=1,#input do
            local inputId = inputVocab.token2id[input[j]]
            encoderInput[i][j + offset] = inputId
        end
        for j=1,#output do
            local outputId = outputVocab.token2id[output[j]]
            decoderInput[i][j + 1] = outputId
            decoderOutput[i][j] = outputId
        end
    end

    return {encoderInput=encoderInput, 
            decoderInput=decoderInput, 
            decoderOutput=decoderOutput,
            sizeExamples=#inputs,
            sizePredictions=totalOutputSize,
            outputSizes=outputSizes}
end


function data.chunkDataset(dataset, numChunks)
    local encInChunks = torch.chunk(dataset.encoderInput, numChunks)
    local decInChunks = torch.chunk(dataset.decoderInput, numChunks)
    local decOutChunks = torch.chunk(dataset.decoderOutput, numChunks)
    local outSizeChunks = torch.chunk(dataset.outputSizes, numChunks)
    local datasets = {}

    for i=1,#decOutChunks do
        datasets[i] = {encoderInput=encInChunks[i], 
                       decoderInput=decInChunks[i],
                       decoderOutput=decOutChunks[i],
                       sizeExamples=encInChunks[i]:size(1),
                       sizePredictions=outSizeChunks[i]:sum()}
    end
    return datasets
end

function data.batches(dataset, batchSize, isGPU)
    
    local X = dataset.encoderInput
    local Yin = dataset.decoderInput
    local Yout = dataset.decoderOutput
    

    local batchX = torch.LongTensor(batchSize, X:size(2)):zero()
    local batchYin = torch.LongTensor(batchSize, Yin:size(2)):zero()
    local batchYout = torch.LongTensor(batchSize, Yout:size(2)):zero()

    if isGPU then
        batchX = batchX:cuda()
        batchYin = batchYin:cuda()
        batchYout = batchYout:cuda()
    end

    local i=1

    local batch_num = 0
    return function()
        if i > X:size(1) then return nil end
        local b = math.min(i + batchSize - 1, X:size(1))

        batch_num = batch_num + 1
        batchX:zero()
        batchYin:zero()
        batchYout:zero()
        batchX[{{1,b-i+1},{}}]:copy(X[{{i,b},{}}])
        batchYin[{{1,b-i+1},{}}]:copy(Yin[{{i,b},{}}])
        batchYout[{{1,b-i+1},{}}]:copy(Yout[{{i,b},{}}])
        
        local position = b
        i = b + 1
        
        return {encoderInput=batchX, decoderInput=batchYin,
                decoderOutput=batchYout,
                num=batch_num, position=position}

    end

end

