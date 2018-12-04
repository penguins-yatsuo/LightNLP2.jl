import Merlin, Merlin.CUDA

const CPU = -1

struct DeviceConfig 
    device_id::Int
end

DEVICE = DeviceConfig(CPU)

macro setdevice(device)
    return Expr(:call, :device_config, esc(device))
end

macro device(ex)
    return Expr(:call, :todevice!, esc(ex), Expr(:call, :device_config))
end

macro host(ex)
    return Expr(:call, :todevice!, esc(ex), CPU)
end

function device_config(device::Union{Int, Nothing}=nothing)
    if device != nothing
        global DEVICE = DeviceConfig(device)
        if device != CPU
            Merlin.CUDA.setdevice(DEVICE.device_id)
        end
    end
    DEVICE.device_id
end

function Merlin.todevice!(o::Nothing, device::Int)
    nothing
end

function Merlin.todevice!(linear::Merlin.Linear, device::Int)
    todevice!(linear.W, device)
    todevice!(linear.b, device)
    linear
end

function Merlin.todevice!(conv::Merlin.Conv1d, device::Int)
    todevice!(conv.W, device)
    todevice!(conv.b, device)
    conv
end

function Merlin.todevice!(lstm::Merlin.LSTM, device::Int)
    foreach(lstm.weights) do weight
        todevice!(weight, device)
    end
    lstm
end

function Merlin.todevice!(layers::Vector, device::Int)
    foreach(layers) do layer
        todevice!(layer, device)
    end
    layers
end

function Merlin.todevice!(layers::Dict, device::Int)
    foreach(values(layers)) do layer
        todevice!(layer, device)
    end
    layers
end


mutable struct Embedding
    W::Merlin.Var
end

function Embedding(w::Matrix{T}; trainable::Bool=true) where T
    if trainable
        Embedding(Merlin.parameter(w))
    else
        Embedding(Merlin.Var(w))
    end
end

function (f::Embedding)(x::Merlin.Var)
    Merlin.lookup(f.W, x)
end

function Merlin.todevice!(embedding::Embedding, device::Int)
    Merlin.todevice!(embedding.W, device)
    embedding
end


function vsize(var::Merlin.Var)
    size(var, 1)
end

function argmax(v::Merlin.Var, dims::Int=1)
    argmax(v.data, dims)
end

function argmax(x::Array, dims::Int=1)
    maxval, maxidx = findmax(x, dims=dims)
    cat(dims=1, map(cart -> cart.I[1], maxidx)...)
end

function Base.string(x::Merlin.Var)
    string("Var",
        " data=", (x.data == nothing ? "nothing" : string(typeof(x.data), size(x.data))),
        " f=", (x.f == nothing ? "nothing" : string(x.f)),
        " args=", (x.args == nothing ? "nothing" : string(typeof(x.args))),
        " grad=", (x.grad == nothing ? "nothing" : string(typeof(x.grad), size(x.grad))))
end




