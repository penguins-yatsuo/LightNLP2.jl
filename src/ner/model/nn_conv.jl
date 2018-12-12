
import Formatting

using Merlin: istrain, todevice!, parameter, Var, data
using Merlin: lookup, max, concat, dropout, relu, softmax, softmax_crossentropy
using Merlin: Linear, Conv1d
using Merlin.CUDA: getdevice

struct ConvNet
    win_c::Int
    win_w::Int
    droprate::Float64
    hidden_dims::Vector{Int}
    out_dim::Int
    L::Dict{String, Any}
end

function ConvNet(args::Dict, embeds_w::Matrix{T}, embeds_c::Matrix{T}, out_dim::Int) where T
    win_c = get!(args, "winsize_c", 2)
    win_w = get!(args, "winsize_w", 5)
    droprate = get!(args, "droprate", 0.1)
    hidden_dims = map(s -> tryparse(Int, s), split(get!(args, "hidden_dims", "128:128"), ":"))
    filter!(x -> !isa(x, Nothing), hidden_dims)

    L = Dict{String, Any}(
        "c_embed" => Embedding(embeds_c; trainable=true),
        "w_embed" => Embedding(embeds_w; trainable=true),
    )

    ConvNet(win_c, win_w, droprate, hidden_dims, out_dim, L)
end

function Base.string(net::ConvNet)
    Formatting.format("Conv <hidden_dims:{1}, winsize_c:{2} winsize_w:{3} droprate:{4:.2f}>",
        string(net.hidden_dims), net.win_c, net.win_w, net.droprate)
end

Merlin.todevice!(net::ConvNet, device::Int) = Merlin.todevice!(net.L, device)

function init_output!(net::ConvNet, out_dim::Int)
    net.out_dim = out_dim
    delete!(net.L, "fc")
end

function (net::ConvNet)(::Type{T}, x::Sample) where T
    c = @device Var(x.c)
    w = @device Var(x.w)

    # embeddings
    c_embed = net.L["c_embed"]
    w_embed = net.L["w_embed"]
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
        @device Linear(T, vsize(h), net.out_dim)
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
