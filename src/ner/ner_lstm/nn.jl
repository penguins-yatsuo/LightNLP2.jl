
struct NN
    g
end

function NN(embeds_w::Matrix{T}, embeds_c::Matrix{T}, ntags::Int;
            nlayers::Int=1, winsize_c::Int=2, droprate::Float64=0.1, bidirectional::Bool=true) where T

    embeds_w = zerograd(embeds_w)
    w = lookup(Node(embeds_w), Node(name="w"))

    embeds_c = zerograd(embeds_c)
    c = lookup(Node(embeds_c), Node(name="c"))
    batchdims_c = Node(name="batchdims_c")
    c = window1d(c, winsize_c, batchdims_c)

    csize = (winsize_c * 2 + 1) * size(embeds_c, 1)
    c = Linear(T, csize, csize)(c)
    c = maximum(c, 2, batchdims_c)

    h = concat(1, w, c)
    batchdims_w = Node(name="batchdims_w")
    hsize = size(embeds_w, 1) + csize
    
    h = LSTM(T, hsize, hsize, nlayers, droprate, bidirectional)(h, batchdims_w)

    h = Linear(T, 2hsize, ntags)(h)
    g = BACKEND(Graph(h))
    NN(g)
end

function (nn::NN)(x::Sample, train::Bool)
    Merlin.CONFIG.train = train
    z = nn.g(x.batchdims_c, x.batchdims_w, Var(x.c), Var(x.w))
    if train
        softmax_crossentropy(Var(x.t), z)
    else
        argmax(z.data, 1)
    end
end
