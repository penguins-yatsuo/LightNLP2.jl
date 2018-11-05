
using Printf: @sprintf

using Merlin: istrain, todevice, parameter, Var
using Merlin: lookup, max, concat, dropout, relu, softmax_crossentropy
using Merlin: Linear, Conv1d

struct ConvNet
    ntags::Int
    nlayers::Int
    winsize_c::Int
    winsize_w::Int
    droprate::Float64
    model::Dict
end

function ConvNet(args::Dict)
    ntags = get!(args, "ntags", 128)
    nlayers = get!(args, "nlayers", 2)
    winsize_c = get!(args, "winsize_c", 2)
    winsize_w = get!(args, "winsize_w", 5)
    droprate = get!(args, "droprate", 0.1)
    ConvNet(ntags, nlayers, winsize_c, winsize_w, droprate, Dict())
end

function ConvNet(; ntags::Int=128, nlayers::Int=2, winsize_c::Int=2, winsize_w::Int=5, droprate::Float64=0.1)
    ConvNet(ntags, nlayers, winsize_c, winsize_w, droprate, Dict())
end

function Base.string(nn::ConvNet)
    @sprintf("%s <ntags:%d nlayers:%d winsize_c:%d winsize_w:%d droprate:%f>",
        "Conv", nn.ntags, nn.nlayers, nn.winsize_c, nn.winsize_w, nn.droprate)
end

function todevice!(nn::ConvNet)
    for (key, m) in pairs(nn.model)
        if isa(m, Conv1d) || isa(m, Linear)
            nn.model[key] = todevice(m)
        end
    end
end

function (nn::ConvNet)(::Type{T}, embeds_c::Matrix{T}, embeds_w::Matrix{T}, x::Sample) where T

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
        h_conv = get!(nn.model, string("h_conv_", string(i)),
            todevice(Conv1d(T, nn.winsize_w * 2 + 1, size(h.data, 1), size(h.data, 1), padding=nn.winsize_w)))
        h = relu(dropout(h_conv(h, x.dims_w), nn.droprate))
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
