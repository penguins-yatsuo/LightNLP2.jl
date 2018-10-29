
struct ConvNet
    ntags::Int
    nlayers::Int
    winsize_c::Int
    winsize_w::Int
    droprate::Float64
    model::Dict
end

function ConvNet(ntags::Int; nlayers::Int=2, winsize_c::Int=2, winsize_w::Int=5, droprate::Float64=0.1) where T
    ConvNet(ntags, nlayers, winsize_c, winsize_w, droprate, Dict())
end

function todevice!(nn::ConvNet)
    for (key, m) in pairs(nn.model)
        if isa(m, Merlin.Conv1d) || isa(m, Merlin.Linear)
            nn.model[key] = todevice(m)
        end
    end
end

function (nn::ConvNet)(::Type{T}, embeds_c::Matrix{T}, embeds_w::Matrix{T}, x::Sample) where T
    c = Merlin.todevice(parameter(lookup(embeds_c, x.c)))
    w = Merlin.todevice(parameter(lookup(embeds_w, x.w)))

    # character conv
    c_conv = get!(nn.model, "c_conv", 
        todevice(Merlin.Conv1d(T, nn.winsize_c * 2 + 1, size(c.data, 1), size(w.data, 1), padding=nn.winsize_c)))    
    c = max(c_conv(c, x.batchdims_c), x.batchdims_c)

    # concatinate word and char
    h = concat(1, w, c)

    # hidden layers
    for i in 1:nn.nlayers
        h_conv = get!(nn.model, string("h_conv_", string(i)), 
            todevice(Merlin.Conv1d(T, nn.winsize_w * 2 + 1, size(h.data, 1), size(h.data, 1), padding=nn.winsize_w)))
        h = relu(dropout(h_conv(h, x.batchdims_w), nn.droprate))
    end

    # full connect
    fc = get!(nn.model, "fc", todevice(Linear(T, size(h.data, 1), nn.ntags)))
    o = relu(fc(h))
      
    # result
    if Merlin.istrain()
        softmax_crossentropy(Merlin.todevice(Merlin.Var(x.t)), o)
    else
        argmax(o)
    end
end
