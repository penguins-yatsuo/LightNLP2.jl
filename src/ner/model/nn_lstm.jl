
import Formatting

using Merlin: istrain, todevice, todevice!, parameter, Var
using Merlin: lookup, max, concat, relu, softmax, softmax_crossentropy
using Merlin: Linear, Conv1d, LSTM

struct LstmNet
    hidden_dims::Vector{Int}
    ntags::Int
    winsize_c::Int
    droprate::Float64
    bidirectional::Bool
    L::Dict
end

function LstmNet(args::Dict)
    ntags = get!(args, "ntags", 128)
    winsize_c = get!(args, "winsize_c", 2)
    droprate = get!(args, "droprate", 0.1)
    bidirectional = get!(args, "bidirectional", true)
    hidden_dims = map(s -> tryparse(Int, s), split(get!(args, "hidden_dims", "128:128"), ":"))
    filter!(x -> !isa(x, Nothing), hidden_dims)
    LstmNet(hidden_dims, ntags, winsize_c, droprate, bidirectional, Dict())
end

function Base.string(net::LstmNet)
    Formatting.format("LSTM <hidden_dims:{1} ntags:{2} winsize_c:{3} droprate:{4:.2f} bidirectional:{5}>",
        string(net.hidden_dims), net.ntags, net.winsize_c, net.droprate, string(net.bidirectional))
end

function Merlin.todevice!(net::LstmNet)
    for (key, m) in pairs(net.L)
        if isa(m, Conv1d) || isa(m, Linear) || isa(m, LSTM)
            net.L[key] = todevice!(m)
        end
    end
end

function (net::LstmNet)(::Type{T}, embeds_c::Matrix{T}, embeds_w::Matrix{T}, x::Sample) where T
    c = todevice(parameter(lookup(embeds_c, x.c)))
    w = todevice(parameter(lookup(embeds_w, x.w)))

    # character conv
    c_conv = get!(net.L, "c_conv",
        todevice!(Conv1d(T, net.winsize_c * 2 + 1, size(c.data, 1), size(w.data, 1), padding=net.winsize_c)))
    c = max(c_conv(c, x.dims_c), x.dims_c)

    # concatinate word and char
    h = concat(1, w, c)

    # hidden layers
    h_lstm = get!(net.L, "h_lstm",
        todevice!(LSTM(T, size(h.data, 1), last(net.hidden_dims), length(net.hidden_dims), net.droprate, net.bidirectional)))
    h = relu(h_lstm(h, x.dims_w))

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
