using LibCUDA

struct NN
    g
end

function NN(embeds_w::Matrix{T}, embeds_c::Matrix{T}, ntags::Int) where T
    embeds_w = zerograd(embeds_w)
    w = lookup(Node(embeds_w), Node(name="w"))

    wsize = size(embeds_w, 1)

    embeds_c = zerograd(embeds_c)
    c = lookup(Node(embeds_c), Node(name="c"))
    batchdims_c = Node(name="batchdims_c")
    c = window1d(c, 2, batchdims_c)

    csize = size(embeds_c, 1)
    c = Linear(T, wsize, wsize)(c)
    c = maximum(c, 2, batchdims_c)

    h = concat(1, w, c)
    batchdims_w = Node(name="batchdims_w")

    hsize = (5 * 2 + 1) * 2wsize

    h = window1d(h, 5, batchdims_w)
    # h = dropout(h, 0.3)
    h = Linear(T, hsize, 2wsize)(h)
    h = relu(h)

    # h = window1d(h, 5, batchdims_w)
    # h = dropout(h, 0.3)
    # h = Linear(T, hsize, 2wsize)(h)
    # h = relu(h)

    h = Linear(T, 2wsize, ntags)(h)
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
