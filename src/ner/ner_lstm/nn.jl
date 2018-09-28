using Merlin.CUDA

struct NN
    embeds_w
    embeds_c
    ntags::Int
    nlayers::Int
    winsize_c::Int
    droprate::Float64
    bidirectional::Bool    
    model::Dict
end

function NN(embeds_w::Matrix{T}, embeds_c::Matrix{T}, ntags::Int;
    nlayers::Int=1, winsize_c::Int=2, droprate::Float64=0.1, bidirectional::Bool=true) where T

    setcuda(0)
    NN(embeds_w, embeds_c, ntags, nlayers, winsize_c, droprate, bidirectional, Dict())
end

function deconfigure!(nn::NN)
    if haskey(nn.model, "h_lstm")
        lstm = get(nn.model, "h_lstm", nothing)
        if isa(lstm, LSTM) && lstm.iscuda
            W = lstm.params[1]
            isnothing(W.data) || (W.data = Array(W.data))
            isnothing(W.grad) || (W.grad = Array(W.grad))

            print(string(typeof(W)))
            lstm.params = (W,)
        end
    end
end

function (nn::NN)(::Type{T}, x::Sample, train::Bool) where T
    settrain(train)    

    c = param(lookup(nn.embeds_c, x.c))
    w = param(lookup(nn.embeds_w, x.w))

    # character conv
    c_conv = get!(nn.model, "c_conv", 
        Conv1d(T, nn.winsize_c * 2 + 1, size(c.data, 1), size(w.data, 1), padding=nn.winsize_c))
    c = c_conv(c, x.batchdims_c)
    c = max(c, x.batchdims_c)

    # word and char
    h = concat(1, w, c)

    # hidden layer
    h_lstm = get!(nn.model, "h_lstm", 
        LSTM(T, size(h.data, 1), size(h.data, 1), nn.nlayers, nn.droprate, nn.bidirectional))
    h = h_lstm(h, x.batchdims_w) 
    h = relu(h)

    # output layer    
    o_linear = get!(nn.model, "o_linear", Linear(T, size(h.data, 1), nn.ntags))
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
