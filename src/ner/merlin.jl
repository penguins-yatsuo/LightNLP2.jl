import Merlin

const CPU = -1

macro device(ex)
    return Expr(:call, :todevice, esc(ex), Expr(:call, :getdevice))
end

macro cpu(ex)
    return Expr(:call, :todevice, esc(ex), CPU)
end

macro ondevice(ex)
    return Expr(:call, :todevice!, esc(ex), Expr(:call, :getdevice))
end

macro oncpu(ex)
    return Expr(:call, :todevice!, esc(ex), CPU)
end

function Merlin.todevice(a::Nothing, device::Int)
    nothing
end

function Merlin.todevice(src::Merlin.Linear, device::Int)
    Merlin.Linear(todevice(src.W, device), todevice(src.b, device))
end

function Merlin.todevice(src::Merlin.Conv1d, device::Int)
    Merlin.Conv1d(src.ksize, src.padding, src.stride, src.dilation, todevice(src.W, device), todevice(src.b, device))
end

function Merlin.todevice(src::Merlin.LSTM, device::Int)
    Merlin.LSTM(src.insize, src.hsize, src.nlayers, src.droprate, src.bidir, map(w -> todevice(w, device), src.weights))
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




