
import Formatting

using Merlin: istrain, todevice!, parameter, Var, data
using Merlin: lookup, max, concat, relu, softmax, softmax_crossentropy
using Merlin: Linear, Conv1d, LSTM

struct LstmNet
    hidden_dims::Vector{Int}
    ntags::Int
    win_c::Int
    droprate::Float64
    bidir::Bool
    L::Dict{String, Any}
end

function LstmNet(args::Dict, embeds_w::Matrix{T}, embeds_c::Matrix{T}) where T
    ntags = get!(args, "ntags", 128)
    win_c = get!(args, "winsize_c", 2)
    droprate = get!(args, "droprate", 0.1)
    bidir = get!(args, "bidirectional", true)
    hidden_dims = map(s -> tryparse(Int, s), split(get!(args, "hidden_dims", "128:128"), ":"))
    filter!(x -> !isa(x, Nothing), hidden_dims)

    L = Dict{String, Any}(
        "c_embed" => Embedding(embeds_c; trainable=true),
        "w_embed" => Embedding(embeds_w; trainable=true),
    )

    LstmNet(hidden_dims, ntags, win_c, droprate, bidir, L)
end

function Base.string(net::LstmNet)
    Formatting.format("LSTM <hidden_dims:{1} ntags:{2} winsize_c:{3} droprate:{4:.2f} bidirectional:{5}>",
        string(net.hidden_dims), net.ntags, net.win_c, net.droprate, string(net.bidir))
end

Merlin.todevice!(net::LstmNet, device::Int) = Merlin.todevice!(net.L, device)

function init_output!(net::LstmNet)
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
