struct Sample
    w
    batchdims_w
    c
    batchdims_c
    t
end

function Base.string(x::Sample)   
    string("Sample", 
        " w=", string(size(x.w)), string(x.w), 
        " dim_w=", string(size(x.batchdims_w)), string(x.batchdims_w), 
        " c=", string(size(x.c)), string(x.c), 
        " dim_c=", string(size(x.batchdims_c)), string(x.batchdims_c), 
        " t=", string(size(x.t)), string(x.t))
end

function create_batch(samples::Vector{Sample}, batchsize::Int, n_samples::Int=0; words)
    batches = Sample[]

    if n_samples == 0
        n_samples = length(samples)
    end

    for i = 1:batchsize:n_samples
        range = i:min(i+batchsize-1,n_samples)
        s = samples[range]
        w = cat(dims=2, map(x -> x.w, s)...)
        c = cat(dims=2, map(x -> x.c, s)...)
        batchdims_w = cat(dims=1, map(x -> x.batchdims_w, s)...)
        batchdims_c = cat(dims=1, map(x -> x.batchdims_c, s)...)
        t = s[1].t == nothing ? nothing : cat(dims=1, map(x -> x.t, s)...)

        w = w


        println("w:$w")

        if words != nothing
            text = map(x -> words[x], w)
            println("text:$text")
        end

        println("batchdims_w:$batchdims_w")
        batchdims_w = sort(batchdims_w, rev=true)
        println("batchdims_w:$batchdims_w")
        break
        c = c
        batchdims_c = batchdims_c
        t = t

        push!(batches, Sample(w, batchdims_w, c, batchdims_c, t))
    end
    batches
end


