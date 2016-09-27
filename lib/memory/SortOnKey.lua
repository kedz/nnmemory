local SortOnKey, Parent = torch.class('nn.SortOnKey', 'nn.Module')

function SortOnKey:__init(descending)
    if descending == nil then 
        descending = false
    end

    self.descending = descending
    self.key_sorted = torch.Tensor()
    self.indices_sorted = torch.LongTensor()
    self.memory_sorted = torch.Tensor()
    self.indices_inv = torch.LongTensor()
    self.gradMemory = torch.Tensor()
    self.gradKey = torch.Tensor()
    self.buffer1 = torch.Tensor() -- Memory buffer for inverse sort.

end


function SortOnKey:updateOutput(input)

    assert(type(input) == 'table')
    assert(#input == 2)

    local memory, key = unpack(input)
    assert(memory:dim() == 3)
    assert(key:dim() == 2)
    assert(memory:size(1) == key:size(1))
    assert(memory:size(2) == key:size(2))
    local batchSize = memory:size(2)

    local key_sorted, indices_sorted = torch.sort(
        self.key_sorted, self.indices_sorted, key, 1, self.descending)

    local memory_sorted = self.memory_sorted:resizeAs(memory)
    for b=1,batchSize do
        memory_sorted:select(2,b):index(
            memory:select(2,b), 1, indices_sorted:select(2,b))
    end

    self.output = {memory_sorted, key_sorted}
    return self.output
end

function SortOnKey:updateGradInput(input, gradOutput)

    assert(type(input) == 'table')
    assert(#input == 2)
    local memory, key = unpack(input)

    assert(memory:dim() == 3)
    assert(key:dim() == 2)
    assert(memory:size(1) == key:size(1))
    assert(memory:size(2) == key:size(2))
    assert(type(input) == 'table')

    assert(#gradOutput == 2)
    local gradMemory, gradKey = unpack(gradOutput)

    assert(gradMemory:dim() == 3)
    assert(gradKey:dim() == 2)
    assert(gradMemory:size(1) == gradKey:size(1))
    assert(gradMemory:size(2) == gradKey:size(2))

    assert(gradMemory:size(1) == memory:size(1))
    assert(gradMemory:size(2) == memory:size(2))
    assert(gradMemory:size(3) == memory:size(3))

    local batchSize = memory:size(2)

    local indices_sorted = 
        self.buffer1:resizeAs(gradKey):copy(self.indices_sorted)
    local _, indices_inv = torch.sort(
        self.buffer1, self.indices_inv, indices_sorted, 1)

    local gradMemoryInput = self.gradMemory:resizeAs(gradMemory)
    local gradKeyInput = self.gradKey:resizeAs(gradKey)
    for b=1,batchSize do
        gradMemoryInput:select(2,b):index(
            gradMemory:select(2,b), 1, indices_inv:select(2,b))
        gradKeyInput:select(2,b):index(
            gradKey:select(2,b), 1, indices_inv:select(2,b))
    end

    self.gradInput = {gradMemoryInput, gradKeyInput}
    return self.gradInput

end

