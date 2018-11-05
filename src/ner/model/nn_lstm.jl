using Merlin: istrain, todevice, parameter, Var
using Merlin: lookup, max, concat, relu, softmax, softmax_crossentropy
using Merlin: Linear, Conv1d, LSTM

struct LstmNet
    ntags::Int
    nlayers::Int
    winsize_c::Int
    droprate::Float64
    bidirectional::Bool
    model::Dict
end

function LstmNet(args::Dict)
    ntags = get!(args, "ntags", 128)
    nlayers = get!(args, "nlayers", 2)
    winsize_c = get!(args, "winsize_c", 2)
    bidirectional = get!(args, "bidirectional", true)
    droprate = get!(args, "droprate", 0.1)
    LstmNet(ntags, nlayers, winsize_c, droprate, bidirectional, Dict())
end

function LstmNet(; ntags::Int=128, nlayers::Int=2, winsize_c::Int=2, bidirectional::Bool=true, droprate::Float64=0.1) where T
    LstmNet(ntags, nlayers, winsize_c, droprate, bidirectional, Dict())
end

function Base.string(nn::LstmNet)
    @sprintf("%s <ntags:%d nlayers:%d winsize_c:%d bidirectional:%s droprate:%f>",
        "LSTM", nn.ntags, nn.nlayers, nn.winsize_c, string(nn.bidirectional), nn.droprate)
end

function todevice!(nn::LstmNet)
    for (key, m) in pairs(nn.model)
        if isa(m, Conv1d) || isa(m, Linear) || isa(m, LSTM)
            nn.model[key] = todevice(m)
        end
    end
end

function (nn::LstmNet)(::Type{T}, embeds_c::Matrix{T}, embeds_w::Matrix{T}, x::Sample) where T
    c = todevice(parameter(lookup(embeds_c, x.c)))
    w = todevice(parameter(lookup(embeds_w, x.w)))

    # character conv
    c_conv = get!(nn.model, "c_conv",
        todevice(Conv1d(T, nn.winsize_c * 2 + 1, size(c.data, 1), size(w.data, 1), padding=nn.winsize_c)))
    c = max(c_conv(c, x.dims_c), x.dims_c)

    # concatinate word and char
    h = concat(1, w, c)

    # hidden layers
    for i in 1:nn.nlayers
        h_lstm = get!(nn.model, string("h_lstm_", string(i)),
            todevice(LSTM(T, size(h.data, 1), size(h.data, 1), 1, nn.droprate, nn.bidirectional)))
        h = relu(h_lstm(h, x.dims_w))
    end

    # full connect
    fc = get!(nn.model, "fc", todevice(Linear(T, size(h.data, 1), nn.ntags)))
    o = relu(fc(h))

    # result
    if istrain()
        softmax_crossentropy(todevice(Var(x.t)), o)
    else
        argmax(o), softmax(o)
    end
end
