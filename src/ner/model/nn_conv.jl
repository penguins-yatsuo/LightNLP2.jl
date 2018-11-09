
import Formatting

using Merlin: istrain, todevice, todevice!, parameter, Var
using Merlin: lookup, max, concat, dropout, relu, softmax, softmax_crossentropy
using Merlin: Linear, Conv1d

struct ConvNet
    hidden_dims::Vector{Int}
    ntags::Int
    winsize_c::Int
    winsize_w::Int
    droprate::Float64
    L::Dict
end

function ConvNet(args::Dict)
    ntags = get!(args, "ntags", 128)
    winsize_c = get!(args, "winsize_c", 2)
    winsize_w = get!(args, "winsize_w", 5)
    droprate = get!(args, "droprate", 0.1)
    hidden_dims = map(s -> tryparse(Int, s), split(get!(args, "hidden_dims", "128:128"), ":"))
    filter!(x -> !isa(x, Nothing), hidden_dims)
    ConvNet(hidden_dims, ntags, winsize_c, winsize_w, droprate, Dict())
end

function Base.string(net::ConvNet)
    Formatting.format("Conv <hidden_dims:{1}, ntags:{2} winsize_c:{3} winsize_w:{4} droprate:{5:.2f}>",
        string(net.hidden_dims), net.ntags, net.winsize_c, net.winsize_w, net.droprate)
end

function Merlin.todevice!(net::ConvNet)
    for (key, m) in pairs(net.L)
        if isa(m, Conv1d) || isa(m, Linear)
            net.L[key] = todevice!(m)
        end
    end
end

function (net::ConvNet)(::Type{T}, embeds_c::Matrix{T}, embeds_w::Matrix{T}, x::Sample) where T
    c = todevice(parameter(lookup(embeds_c, x.c)))
    w = todevice(parameter(lookup(embeds_w, x.w)))

    # character conv
    c_conv = get!(net.L, "c_conv",
        todevice!(Conv1d(T, net.winsize_c * 2 + 1, size(c.data, 1), size(w.data, 1), padding=net.winsize_c)))
    c = max(c_conv(c, x.dims_c), x.dims_c)

    # concatinate word and char
    h = concat(1, w, c)

    # hidden layers
    for i in 1:length(net.hidden_dims)
        h_conv = get!(net.L, "h_conv_$i",
            todevice!(Conv1d(T, net.winsize_w * 2 + 1, size(h.data, 1), net.hidden_dims[i], padding=net.winsize_w)))
        h = relu(dropout(h_conv(h, x.dims_w), net.droprate))
    end

    # full connect
    fc = get!(net.L, "fc", todevice!(Linear(T, size(h.data, 1), net.ntags)))
    o = relu(fc(h))

    # result
    if istrain()
        softmax_crossentropy(todevice(Var(x.t)), o)
    else
        p = softmax(o)
        argmax(o), Array(p.data)
    end
end
