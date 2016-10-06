-- Options
local opt = lapp [[
Train and evaluate coupled LSTM encoder/decoder on toy data.
Options:
   --input-vocab     (string)        Encoder vocab path 
   --output-vocab    (string)        Decoder vocab path 
   --train-path      (string)        Training data path
   --test-path       (string)        Test data path 
   --batch-size      (default 50)    Batch size
   --learning-rate   (default .001)  Learning rate
   --dim-size        (default 200)   Dimension of of all embeddings, units.
   --num-layers      (default 2)     Number of stacked lstms.
   --seed            (default 1)     Seed for torch random number generator
   --results-tsv     (string)        Path to write results.
   --progress-bar                    Show progress bar
   --gpu             (default 0)     Which gpu to use. Default is cpu.
]]

-- require torch packages
require 'nn'
require 'rnn'
require 'optim'

local use_GPU = false
if opt.gpu > 0 then use_GPU = true end

if use_GPU then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpu)
    cutorch.manualSeed(opt.seed)
    torch.manualSeed(opt.seed)
    print("running on gpu-" .. opt.gpu)

else
    torch.manualSeed(opt.seed)
    print("running on cpu")
end

-- require my packages
require 'data'
require 'memory'
require 'models'


local inputVocab = data.readVocab(opt.input_vocab, false)
local outputVocab = data.readVocab(opt.output_vocab, true)

local trainData = data.readData(opt.train_path, inputVocab, outputVocab, true)
local testData = data.readData(opt.test_path, inputVocab, outputVocab, true)
local testDataLarge = data.readData(
    opt.test_path_large, inputVocab, outputVocab, true)

optimState = {learningRate=opt.learning_rate}
local model = nn.memory.LSTMTransducer(inputVocab.size, outputVocab.size, 
    opt.dim_size, opt.num_layers, optimState) 

if use_GPU then model:cuda() end

local trainingTimes = {}
local testTimes = {}

local numTrainChunks = math.floor(trainData.sizeExamples / 1000)
print("Splitting training data into " .. numTrainChunks .. " chunks.")
local trainDataChunks = data.chunkDataset(trainData, numTrainChunks)
local lastTrainIndex = 1

for chunk=1,#trainDataChunks do
    local nextTrainIndex = lastTrainIndex + trainDataChunks[chunk].sizeExamples

    print("\nTrain chunk " .. lastTrainIndex .." ..  " .. nextTrainIndex-1)
    lastTrainIndex = nextTrainIndex

    -- TRAINING --

    local trainChunk = trainDataChunks[chunk]
    local trainLoss = 0
    local startTrain = os.clock()
    for batch in data.batches(trainChunk, opt.batch_size, use_GPU) do 
        if opt.progress_bar then
            xlua.progress(batch.position, trainChunk.sizeExamples)
        end
        local err = model:train(
            batch.encoderInput, batch.decoderInput, batch.decoderOutput)
        trainLoss = trainLoss + err
    end
    local stopTrain = os.clock()
    local trainPerpl = torch.exp(trainLoss / trainChunk.sizePredictions)
    print(string.format("Training time %.2f", stopTrain - startTrain))
    print("Training perpl. = ".. trainPerpl)
    table.insert(trainingTimes, stopTrain - startTrain)

    -- TEST (same avg. size as training set) --

    local testLoss = 0
    local startTest = os.clock()
    for batch in data.batches(testData, opt.batch_size, use_GPU) do 
        if opt.progress_bar then
            xlua.progress(batch.position, testData.sizeExamples)
        end
        local err = model:loss(
            batch.encoderInput, batch.decoderInput, batch.decoderOutput)
        testLoss = testLoss + err
    end
    local stopTest = os.clock()
    local testPerpl = torch.exp(testLoss / testData.sizePredictions)
    print(string.format("Test time %.2f", stopTest - startTest))
    print("Test perpl. = "..testPerpl)
    table.insert(testTimes, stopTest - startTest)

    local out = assert(io.open(opt.results_tsv, "a"))
    out:write(opt.seed .. "\t" .. opt.learning_rate .. "\t" .. opt.dim_size .. 
        "\t" .. opt.batch_size .. "\t" .. opt.num_layers .. "\t" .. 
        chunk .. "\t" .. trainPerpl .. "\t" .. testPerpl .. "\n")
    out:close()

end

local avgTrainTime = torch.Tensor(trainingTimes):mean()
local avgTestTime = torch.Tensor(testTimes):mean()
print(string.format("Average training time per epoch = %.2f", avgTrainTime))
print(string.format("Average training time per epoch = %.2f", avgTestTime))
