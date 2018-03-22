using LibCUDA

struct NN
    g
end

function NN(embeds_w::Matrix{T}, embeds_c::Matrix{T}, ntags::Int) where T
    embeds_w = zerograd(embeds_w)
    w = lookup(Node(embeds_w), Node(name="w"))

    embeds_c = zerograd(embeds_c)
    c = lookup(Node(embeds_c), Node(name="c"))
    batchdims_c = Node(name="batchdims_c")
    # c = window1d(c, 2, batchdims_c)
    csize = size(embeds_c, 1)
    c = Linear(T, 5csize, 5csize)(c)
    c = maximum(c, 2, batchdims_c)

    # batchdims_w = Node(name="batchdims_w")
    h = concat(1, w, c)
    hsize = size(embeds_w, 1) + 5csize

    println(hsize)

    for i = 1:2
        h = dropout(h, 0.3)
        h = Conv(T, 5, hsize, hsize, 2, 1)(h)
        h = relu(h)
    end

    h = Linear(T, 5hsize, ntags)(h)
    g = BACKEND(Graph(h))

    NN(g)
end

function (nn::NN)(x::Sample, train::Bool)
    Merlin.CONFIG.train = train
    z = nn.g(x.batchdims_c, Var(x.c), Var(x.w))
    if train
        softmax_crossentropy(Var(x.t), z)
    else
        argmax(z.data, 1)
    end
end
