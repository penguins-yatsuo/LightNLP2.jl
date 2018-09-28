using Merlin.CUDA

struct NN
    embeds_w
    embeds_c
    ntags::Int
    nlayers::Int
    winsize_c::Int
    winsize_w::Int
    droprate::Float64

    model::Dict
end

function NN(embeds_w::Matrix{T}, embeds_c::Matrix{T}, ntags::Int; 
            nlayers::Int=2, winsize_c::Int=2, winsize_w::Int=5, droprate::Float64=0.1) where T
    setcpu()
    NN(embeds_w, embeds_c, ntags, nlayers, winsize_c, winsize_w, droprate, Dict())
end

function (nn::NN)(::Type{T}, x::Sample, train::Bool) where T
    settrain(train)    

    c = param(lookup(nn.embeds_c, x.c); name="c")
    c_dims = Var(x.batchdims_c, name="c_dims")
    w = param(lookup(nn.embeds_w, x.w); name="w")
    w_dims = Var(x.batchdims_w, name="w_dims")

    # character conv
    c_conv = get!(nn.model, "c_conv", 
        Conv1d(T, nn.winsize_c, size(c.data, 1), size(w.data, 1), padding=Int((nn.winsize_c - 1) / 2)))
    c = c_conv(c, c_dims)
    c = max(c, c_dims)

    # word and char
    h = concat(1, w, c)

    # hidden layers
    for i in 1:nn.nlayers
        h_conv = get!(nn.model, string("h_conv_", string(i)), 
            Conv1d(T, nn.winsize_w, size(h.data, 1), size(h.data, 1), padding=Int((nn.winsize_w - 1) / 2)))
        h = h_conv(h, w_dims)
        h = dropout(h, nn.droprate)
        h = relu(h)
    end
    
    o_linear = get!(nn.model, "o_linear", Linear(T, size(h.data, 1), nn.ntags))
    o = o_linear(h)
    o = relu(o)
      
    if train
        t = Var(x.t; name="tag")
        softmax_crossentropy(t, o)
    else
        argmax(o)
    end
end


function argmax(v::Var)
    x = isa(v.data, CuArray) ? Array(v.data) : v.data 
    maxval, maxidx = findmax(v.data, dims=1)
    cat(dims=1, map(cart -> cart.I[1], maxidx)...)
end
