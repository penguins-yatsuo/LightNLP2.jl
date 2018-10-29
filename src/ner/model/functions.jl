
function argmax(v::Var)
    x = Array(v.data)
    maxval, maxidx = findmax(x, dims=1)
    cat(dims=1, map(cart -> cart.I[1], maxidx)...)
end

function Base.string(x::Var)   
    string("Var",
        " data=", (isnothing(x.data) ? "nothing" : string(typeof(x.data), size(x.data))),
        " f=", (isnothing(x.f) ? "nothing" : string(x.f)),
        " args=", (isnothing(x.args) ? "nothing" : string(typeof(x.args))),
        " grad=", (isnothing(x.grad) ? "nothing" : string(typeof(x.grad), size(x.grad))))
end
