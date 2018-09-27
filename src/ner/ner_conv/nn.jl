# using LibCUDA

struct NN
    embeds_w
    embeds_c
    ntags::Int
    nlayers::Int
    winsize_c::Int
    winsize_w::Int
    droprate::Float64
end

function NN(embeds_w::Matrix{T}, embeds_c::Matrix{T}, ntags::Int; 
            nlayers::Int=2, winsize_c::Int=2, winsize_w::Int=5, droprate::Float64=0.1) where T
    NN(embeds_w, embeds_c, ntags, nlayers, winsize_c, winsize_w, droprate)
end

function char_conv(::Type{T}, c::Var, dims::Var, outsize::Int, window::Int) where T

    insize = size(c.data, 1)
    pad = Int((window - 1) / 2)

    h = Conv1d(T, window, insize, outsize, padding=pad)(c, dims)
    h = max(h, dims)

    h 
end

function (nn::NN)(::Type{T}, x::Sample, train::Bool) where T
    settrain(train)    

    # setcuda(0)
    c = param(lookup(nn.embeds_c, x.c); name="c")
    c_dims = Var(x.batchdims_c, name="c_dims")
    w = param(lookup(nn.embeds_w, x.w); name="w")
    w_dims = Var(x.batchdims_w, name="w_dims")

    wsize = size(w.data, 1)

    # character conv    
    c = char_conv(T, c, c_dims, wsize, nn.winsize_c)

    # word and char
    wc = concat(1, w, c)  
    wcsize = size(wc.data, 1)

    # hidden layer
    h = wc
    hsize = 1024
    pad = Int((nn.winsize_w - 1) / 2)

    for i in 1:nn.nlayers
        h = Conv1d(T, nn.winsize_w, (i == 1 ? wcsize : hsize), hsize, padding=pad)(h, w_dims)
        h = dropout(h, nn.droprate)
        h = relu(h)
    end
    
    h = Linear(T, hsize, nn.ntags)(h)
    pred = relu(h)
      
    if train
        t = Var(x.t; name="tag")
        softmax_crossentropy(t, pred)
    else
        argmax(pred)
    end
end

function argmax(v::Var)
    maxval, maxidx = findmax(v.data, dims=1)
    cat(dims=1, map(cart -> cart.I[1], maxidx)...)
end
