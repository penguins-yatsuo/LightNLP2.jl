
import Formatting

using Merlin: istrain, todevice!, parameter, Var, data
using Merlin: lookup, max, concat, dropout, relu, softmax, softmax_crossentropy
using Merlin: Linear, Conv1d
using Merlin.CUDA: getdevice

struct ConvNet
    hidden_dims::Vector{Int}
    ntags::Int
    win_c::Int
    win_w::Int
    droprate::Float64
    L::Dict
end

function ConvNet(args::Dict)
    ntags = get!(args, "ntags", 128)
    win_c = get!(args, "winsize_c", 2)
    win_w = get!(args, "winsize_w", 5)
    droprate = get!(args, "droprate", 0.1)
    hidden_dims = map(s -> tryparse(Int, s), split(get!(args, "hidden_dims", "128:128"), ":"))
    filter!(x -> !isa(x, Nothing), hidden_dims)

    ConvNet(hidden_dims, ntags, win_c, win_w, droprate, Dict())
end

function Base.string(net::ConvNet)
    Formatting.format("Conv <hidden_dims:{1}, ntags:{2} winsize_c:{3} winsize_w:{4} droprate:{5:.2f}>",
        string(net.hidden_dims), net.ntags, net.win_c, net.win_w, net.droprate)
end

Merlin.todevice!(net::ConvNet, device::Int) = Merlin.todevice!(net.L, device)

function (net::ConvNet)(::Type{T}, embeds_w::Matrix{T}, embeds_c::Matrix{T}, x::Sample) where T
    c = @device Var(x.c)
    w = @device Var(x.w)

    # embeddings
    c_embed = get!(net.L, "c_embed") do
        @device Embedding(embeds_c)
    end
    w_embed = get!(net.L, "w_embed") do 
        @device Embedding(embeds_w)
    end
    c = c_embed(c)
    w = w_embed(w)

    # character conv
    c_conv = get!(net.L, "c_conv") do 
        @device Conv1d(T, 2(net.win_c) + 1, vsize(c), vsize(w), padding=net.win_c)
    end
    c = max(c_conv(c, x.dims_c), x.dims_c)

    # concatinate word and char
    h = concat(1, w, c)

    # hidden layers
    for i in 1:length(net.hidden_dims)
        h_conv = get!(net.L, "h_conv_$i") do 
            @device Conv1d(T, 2(net.win_w) + 1, vsize(h), net.hidden_dims[i], padding=net.win_w)
        end        
        h = relu(dropout(h_conv(h, x.dims_w), net.droprate))
    end

    # full connect
    fc = get!(net.L, "fc") do
        @device Linear(T, vsize(h), net.ntags)
    end    
    o = relu(fc(h))

    # result
    if istrain()
        t = @device Var(x.t)
        softmax_crossentropy(t, o)
    else
        softmax(o)
    end
end
