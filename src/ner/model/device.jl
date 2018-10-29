

function todevice(f::Merlin.Conv1d)
    Merlin.Conv1d(f.ksize, f.padding, f.stride, f.dilation, Merlin.todevice(f.W), Merlin.todevice(f.b))
end

function todevice(f::Merlin.Linear)
    Merlin.Linear(Merlin.todevice(f.W), Merlin.todevice(f.b))
end

function todevice(f::Merlin.LSTM)
    Merlin.LSTM(f.insize, f.hsize, f.nlayers, f.droprate, f.bidir, map(w -> Merlin.todevice(w), f.weights))
end