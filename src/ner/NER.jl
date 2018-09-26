module NER
using Merlin
using ProgressMeter
using HDF5

module Convolution


# const BACKEND = CPUBackend()

include("BIOES.jl")
include("ner_conv/decoder.jl")
include("ner_conv/nn.jl")

end

module Lstm

# const BACKEND = CPUBackend()

include("BIOES.jl")
include("ner_lstm/decoder.jl")
include("ner_lstm/nn.jl")

end

end
