require 'torch'
--require 'torchx'
--dpnn.version = dpnn.version or 0
--assert(dpnn.version > 1, "Please update dpnn : luarocks install dpnn")

-- create global memory table:
memory = {}
memory.version = 0.1

unpack = unpack or table.unpack



torch.include('memory', 'LinearAssociativeMemoryReader.lua')

-- priority queue modules
--torch.include('memory', 'PriorityQueueSimpleEncoder.lua')
--torch.include('memory', 'PriorityQueueSimpleAttentionEncoder.lua')
torch.include('memory', 'PriorityQueueSimpleDecoder.lua')

torch.include('memory', 'SortOnKey.lua')
torch.include('memory', 'LinearMemoryWriter.lua')
torch.include('memory', 'BilinearAttentionMemoryWriter.lua')
torch.include('memory', 'MemoryCell.lua')
torch.include('memory', 'CoupledLSTM.lua')
torch.include('memory', 'GradientTests.lua')
torch.include('memory', 'CudaTests.lua')

-- -- recurrent modules
-- torch.include('rnn', 'LookupTableMaskZero.lua')
-- torch.include('rnn', 'MaskZero.lua')
-- torch.include('rnn', 'TrimZero.lua')
-- torch.include('rnn', 'AbstractRecurrent.lua')
-- torch.include('rnn', 'Recurrent.lua')
-- torch.include('rnn', 'LSTM.lua')
-- torch.include('rnn', 'FastLSTM.lua')
-- torch.include('rnn', 'GRU.lua')
-- torch.include('rnn', 'Recursor.lua')
-- torch.include('rnn', 'Recurrence.lua')
-- torch.include('rnn', 'NormStabilizer.lua')
--
-- -- sequencer modules
-- torch.include('rnn', 'AbstractSequencer.lua')
-- torch.include('rnn', 'Repeater.lua')
-- torch.include('rnn', 'Sequencer.lua')
-- torch.include('rnn', 'BiSequencer.lua')
-- torch.include('rnn', 'BiSequencerLM.lua')
-- torch.include('rnn', 'RecurrentAttention.lua')
--
-- -- sequencer + recurrent modules
-- torch.include('rnn', 'SeqLSTM.lua')
-- torch.include('rnn', 'SeqLSTMP.lua')
-- torch.include('rnn', 'SeqGRU.lua')
-- torch.include('rnn', 'SeqReverseSequence.lua')
-- torch.include('rnn', 'SeqBRNN.lua')
--
-- -- recurrent criterions:
-- torch.include('rnn', 'SequencerCriterion.lua')
-- torch.include('rnn', 'RepeaterCriterion.lua')
-- torch.include('rnn', 'MaskZeroCriterion.lua')

-- prevent likely name conflicts
nn.memory = memory
