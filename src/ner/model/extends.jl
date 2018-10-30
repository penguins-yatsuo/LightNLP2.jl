import Merlin

function Merlin.todevice(f::Merlin.Conv1d)
    Merlin.Conv1d(f.ksize, f.padding, f.stride, f.dilation, Merlin.todevice(f.W), Merlin.todevice(f.b))
end

function Merlin.todevice(f::Merlin.Linear)
    Merlin.Linear(Merlin.todevice(f.W), Merlin.todevice(f.b))
end

function Merlin.todevice(f::Merlin.LSTM)
    Merlin.LSTM(f.insize, f.hsize, f.nlayers, f.droprate, f.bidir, map(w -> Merlin.todevice(w), f.weights))
end

function Base.string(x::Merlin.Var)   
    string("Var",
        " data=", (x.data == nothing ? "nothing" : string(typeof(x.data), size(x.data))),
        " f=", (x.f == nothing ? "nothing" : string(x.f)),
        " args=", (x.args == nothing ? "nothing" : string(typeof(x.args))),
        " grad=", (x.grad == nothing ? "nothing" : string(typeof(x.grad), size(x.grad))))
end
