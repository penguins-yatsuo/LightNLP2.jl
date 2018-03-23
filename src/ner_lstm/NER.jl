module NER

using Merlin
using ProgressMeter
using HDF5

const BACKEND = CPUBackend()

include("BIOES.jl")
include("decoder.jl")
include("nn.jl")

end
