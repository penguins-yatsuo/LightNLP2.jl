
import Formatting

using Merlin: istrain, todevice!, parameter, Var, data
using Merlin: lookup, max, concat, relu, softmax, softmax_crossentropy
using Merlin: Linear, Conv1d, LSTM

struct LstmNet
    win_c::Int
    bidir::Bool
    droprate::Float64
    hidden_dims::Vector{Int}
    out_dim::Int
    L::Dict{String, Any}
end

function LstmNet(args::Dict, embeds_w::Matrix{T}, embeds_c::Matrix{T}, out_dim::Int) where T
    win_c = get!(args, "winsize_c", 2)
    droprate = get!(args, "droprate", 0.1)
    bidir = get!(args, "bidirectional", true)
    hidden_dims = map(s -> tryparse(Int, s), split(get!(args, "hidden_dims", "128:128"), ":"))
    filter!(x -> !isa(x, Nothing), hidden_dims)

    L = Dict{String, Any}(
        "c_embed" => Embedding(embeds_c; trainable=true),
        "w_embed" => Embedding(embeds_w; trainable=true),
    )

    LstmNet(win_c, bidir, droprate, hidden_dims, out_dim, L)
end

function Base.string(net::LstmNet)
    Formatting.format("LSTM <hidden_dims:{1} winsize_c:{2} droprate:{3:.2f} bidirectional:{4}>",
        string(net.hidden_dims), net.win_c, net.droprate, string(net.bidir))
end

Merlin.todevice!(net::LstmNet, device::Int) = Merlin.todevice!(net.L, device)

function init_output!(net::LstmNet, out_dim::Int)
    net.out_dim = out_dim
    delete!(net.L, "fc")
end

function (net::LstmNet)(::Type{T}, x::Sample) where T
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

    # hidden layer
    # Merlin.LSTMでnlayersに2以上を設定すると実行時エラーが発生する
    # 代替の実装として1層のLSTMを重ねる
    for i in 1:length(net.hidden_dims)
        h_lstm = get!(net.L, "h_lstm_$i") do 
            @device LSTM(T, vsize(h), net.hidden_dims[i], 1, net.droprate, net.bidir)
        end        
        h = relu(h_lstm(h, x.dims_w))
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
