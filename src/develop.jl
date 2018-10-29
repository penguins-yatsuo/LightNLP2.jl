

function Base.string(x::Merlin.Var)   
    string("Var",
        " data=", (isnothing(x.data) ? "nothing" : string(typeof(x.data), size(x.data))),
        " f=", (isnothing(x.f) ? "nothing" : string(x.f)),
        " args=", (isnothing(x.args) ? "nothing" : string(typeof(x.args))),
        " grad=", (isnothing(x.grad) ? "nothing" : string(typeof(x.grad), size(x.grad))))
end

