export Sample

using Random

struct Sample
    w::Matrix{Int}
    dims_w::Vector{Int}
    c::Matrix{Int}
    dims_c::Vector{Int}
    t::Vector{Int}
end

Base.length(x::Sample) = length(x.dims_w)
word_length(x::Sample) = sum(x.dims_w)
word_length(xs::Vector{Sample}) = sum(n_word, xs)

function Base.string(x::Sample)
    string("Sample",
        " w=", string(size(x.w)), string(x.w),
        " dims_w=", string(size(x.dims_w)), string(x.dims_w),
        " c=", string(size(x.c)), string(x.c),
        " dims_c=", string(size(x.dims_c)), string(x.dims_c),
        " t=", string(size(x.t)), string(x.t))
end

mutable struct SampleIterater
    samples
    batchsize
    n_samples
    sort
    channel
end

function SampleIterater(samples, batchsize::Int, n_samples::Int; shuffle::Bool=false, sort::Bool=false)
    (n_samples < 1 || n_samples > length(samples)) && (n_samples = length(samples))
    shuffle && Random.shuffle!(samples)
    SampleIterater(samples, batchsize, n_samples, sort, nothing)
end

function init!(iter::SampleIterater)
    iter.channel = Channel{Sample}(8)
    offsets = range(1, step=iter.batchsize, length=length(iter))
    task = @async foreach(i -> put!(iter.channel, batch_samples(iter, i)), offsets)
    bind(iter.channel, task)

    # return first element
    take!(iter.channel)
end

function next!(iter::SampleIterater)
    (isopen(iter.channel) || isready(iter.channel)) ? take!(iter.channel) : nothing
end

function Base.iterate(iter::SampleIterater, (el, i)=(init!(iter), 1))
    (el != nothing) ? (el, (next!(iter), i + iter.batchsize)) : nothing
end

Base.length(iter::SampleIterater) = Int(ceil(iter.n_samples / iter.batchsize))
Base.eltype(iter::SampleIterater) = Sample

function batch_samples(iter::SampleIterater, offset::Int)
    (offset > iter.n_samples || offset > length(iter.samples)) && return nothing

    ss = iter.samples[offset:min(offset + (iter.batchsize - 1), iter.n_samples)]
    iter.sort && sort!(ss, rev=true, by=(s -> sum(s.dims_w)))

    w = cat(dims=2, map(x -> x.w, ss)...)
    c = cat(dims=2, map(x -> x.c, ss)...)
    dims_w = cat(dims=1, map(x -> x.dims_w, ss)...)
    dims_c = cat(dims=1, map(x -> x.dims_c, ss)...)
    t = (ss[1].t == nothing) ? nothing : cat(dims=1, map(x -> x.t, ss)...)

    Sample(w, dims_w, c, dims_c, t)
end
