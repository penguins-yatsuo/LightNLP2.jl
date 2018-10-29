
struct LstmNet
    ntags::Int
    nlayers::Int
    winsize_c::Int
    droprate::Float64
    bidirectional::Bool 
    model::Dict
end

function LstmNet(ntags::Int;
        nlayers::Int=2, winsize_c::Int=2, bidirectional::Bool=true, droprate::Float64=0.1) where T

    LstmNet(ntags, nlayers, winsize_c, droprate, bidirectional, Dict())
end


function todevice!(nn::LstmNet)
    for (key, m) in pairs(nn.model)
        if isa(m, Merlin.Conv1d) || isa(m, Merlin.Linear) || isa(m, Merlin.LSTM)
            nn.model[key] = todevice(m)
        end
    end
end

function (nn::LstmNet)(::Type{T}, embeds_c::Matrix{T}, embeds_w::Matrix{T}, x::Sample) where T
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
        h_lstm = get!(nn.model, string("h_lstm_", string(i)), 
            todevice(Merlin.LSTM(T, size(h.data, 1), size(h.data, 1), 1, nn.droprate, nn.bidirectional)))
        h = relu(h_lstm(h, x.batchdims_w))
    end

    # full connect
    fc = get!(nn.model, "fc", todevice(Linear(T, size(h.data, 1), nn.ntags)))
    o = relu(fc(h))
      
    # result
    if Merlin.istrain()
        softmax_crossentropy(Merlin.todevice(Var(x.t)), o)
    else
        argmax(o)
    end
end
