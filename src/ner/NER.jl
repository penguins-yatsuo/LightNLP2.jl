module NER

module Convolution

using Merlin
using ProgressMeter
using HDF5

const BACKEND = CPUBackend()

include("ner_conv/BIOES.jl")
include("ner_conv/decoder.jl")
include("ner_conv/nn.jl")

end

module Lstm

using Merlin
using ProgressMeter
using HDF5

const BACKEND = CPUBackend()

include("ner_lstm/BIOES.jl")
include("ner_lstm/decoder.jl")
include("ner_lstm/nn.jl")

end

end
