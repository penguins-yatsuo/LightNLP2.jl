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

function (nn::NN)(::Type{T}, x::Sample, train::Bool) where T
    settrain(train)    

    c = param(lookup(nn.embeds_c, x.c); name="lookup")
    w = param(lookup(nn.embeds_w, x.w); name="lookup")

    wsize = 100
    hsize = 1024
    
    h = Linear(T, wsize, hsize)(w)
    h = relu(h)

    h = Linear(T, hsize, nn.ntags)(h)
    pred = relu(h)
      
    if train
        t = Var(x.t; name="tag")
        out = softmax_crossentropy(t, pred)
    else
        println(string(x))
        println(string(pred))
      
        println("pred", string(size(pred.data)), string(pred.data))

        maxval, maxidx = findmax(pred.data, dims=1)

        println("maxval", string(size(maxval)), string(maxval))
        println("maxidx", string(size(maxidx)), string(maxidx))

        for i in 1:sum(x.batchdims_w)
            println(maxval[i], maxidx[i], view(pred.data, i, 1:5))
        end
        
        argmaxidx = cat(dims=1, map(cart -> cart.I[1], maxidx)...)

        println("argmaxidx", string(size(argmaxidx)), string(argmaxidx))


        
        out = fill(1, sum(x.batchdims_w))
    end

    out
end

function argmax(v::Var)

    maxval, maxidx = findmax(v.data, dims=1)
    argmaxidx = cat(dims=1, map(cart -> cart.I[1], maxidx)...)



end
