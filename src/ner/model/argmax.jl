
function argmax(v::Merlin.Var)
    x = Array(v.data)
    maxval, maxidx = findmax(x, dims=1)
    cat(dims=1, map(cart -> cart.I[1], maxidx)...)
end

