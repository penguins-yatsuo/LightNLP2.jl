module Convolution

using Merlin, Merlin.CUDA
using ..NER

struct NN
    embeds_w
    embeds_c
    ntags::Int
    nlayers::Int
    winsize_c::Int
    winsize_w::Int
    droprate::Float64
    use_gpu::Bool
    layers::Dict
end

function NN(embeds_w::Matrix{T}, embeds_c::Matrix{T}, ntags::Int; 
            nlayers::Int=2, winsize_c::Int=2, winsize_w::Int=5, droprate::Float64=0.1, use_gpu::Bool=false) where T
    if use_gpu
        setcuda(0)
    else
        setcpu()
    end
    NN(embeds_w, embeds_c, ntags, nlayers, winsize_c, winsize_w, droprate, use_gpu, Dict())
end

function (nn::NN)(::Type{T}, x::Sample, train::Bool) where T
    settrain(train)    

    c = param(lookup(nn.embeds_c, x.c))
    w = param(lookup(nn.embeds_w, x.w))

    # character conv
    c_conv = get!(nn.layers, "c_conv", 
        Conv1d(T, nn.winsize_c * 2 + 1, size(c.data, 1), size(w.data, 1), padding=nn.winsize_c))
    c = c_conv(c, x.batchdims_c)
    c = max(c, x.batchdims_c)

    # word and char
    h = concat(1, w, c)

    # hidden layers
    for i in 1:nn.nlayers
        h_conv = get!(nn.layers, string("h_conv_", string(i)), 
            Conv1d(T, nn.winsize_w * 2 + 1, size(h.data, 1), size(h.data, 1), padding=nn.winsize_w))
        h = h_conv(h, x.batchdims_w)
        h = dropout(h, nn.droprate)
        h = relu(h)
    end
    
    o_linear = get!(nn.layers, "o_linear", Linear(T, size(h.data, 1), nn.ntags))
    o = o_linear(h)
    o = relu(o)
      
    if train
        softmax_crossentropy(Var(x.t), o)
    else
        argmax(o)
    end
end


function argmax(v::Var)
    x = isa(v.data, CuArray) ? Array(v.data) : v.data 
    maxval, maxidx = findmax(x, dims=1)
    cat(dims=1, map(cart -> cart.I[1], maxidx)...)
end

end # module Convolution