using LibCUDA

struct NN
    g
end

function NN(embeds_w::Matrix{T}, embeds_c::Matrix{T}, ntags::Int; 
            nlayers::Int=2, winsize_c::Int=2, winsize_w::Int=5, droprate::Float64=0.1) where T

    embeds_w = zerograd(embeds_w)
    w = lookup(Node(embeds_w), Node(name="w"))

    wsize = size(embeds_w, 1)

    embeds_c = zerograd(embeds_c)
    c = lookup(Node(embeds_c), Node(name="c"))
    batchdims_c = Node(name="batchdims_c")
    c = window1d(c, winsize_c, batchdims_c)

    csize = (winsize_c * 2 + 1) * size(embeds_c, 1)
    c = Linear(T, csize, csize)(c)
    c = maximum(c, 2, batchdims_c)

    h = concat(1, w, c)
    batchdims_w = Node(name="batchdims_w")

    hsize = wsize + csize
    wsize = (winsize_w * 2 + 1) * hsize

    for i in 1:nlayers
        h = Node(window1d, h, winsize_w, batchdims_w)
        h = Node(dropout, h, droprate)
        h = Linear(T, wsize, hsize)(h)
        h = relu(h)    
    end

    h = Linear(T, hsize, ntags)(h)
    g = BACKEND(Graph(h))

    NN(g)
end

function (nn::NN)(x::Sample, train::Bool)
    Merlin.CONFIG.train = train
    z = nn.g(x.batchdims_c, x.batchdims_w, Var(x.c), Var(x.w))
    #z = nn.g(x.batchdims_c, Var(x.c), Var(x.w))
    if train
        softmax_crossentropy(Var(x.t), z)
    else
        argmax(z.data, 1)
    end
end
