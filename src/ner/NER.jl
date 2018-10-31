module NER

using Dates: format, now

macro timestr()
    return :( format(now(), "yyyy-mm-dd HH:MM:SS") )
end

include("BIOES.jl")
include("dataset.jl")
include("vocab.jl")
# include("sample.jl")
include("decoder.jl")
include("model/argmax.jl")
include("model/extends.jl")
include("model/nn_conv.jl")
include("model/nn_lstm.jl")

end # module NER


