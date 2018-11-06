import Merlin

function Merlin.todevice(x::Nothing)
    nothing
end

function Merlin.todevice!(f::Merlin.Conv1d)
    Merlin.todevice!(f.W)
    Merlin.todevice!(f.b)
    f
end

function Merlin.todevice!(f::Merlin.Linear)
    Merlin.todevice!(f.W)
    Merlin.todevice!(f.b)
    f
end

function Merlin.todevice!(f::Merlin.LSTM)
    map(w -> Merlin.todevice!(w), f.weights)
    f
end

function Base.string(x::Merlin.Var)
    string("Var",
        " data=", (x.data == nothing ? "nothing" : string(typeof(x.data), size(x.data))),
        " f=", (x.f == nothing ? "nothing" : string(x.f)),
        " args=", (x.args == nothing ? "nothing" : string(typeof(x.args))),
        " grad=", (x.grad == nothing ? "nothing" : string(typeof(x.grad), size(x.grad))))
end
