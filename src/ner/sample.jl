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

function load_samples(path::String, words::Vector{String}, chars::Vector{Char}, tags::Vector{String})

    word_dic = Dict(words[i] => i for i in 1:length(words))
    char_dic = Dict(chars[i] => i for i in 1:length(chars))
    tag_dic = Dict(tags[i] => i for i in 1:length(tags))

    unknown_w = length(words)
    unknown_c = length(chars)
    unknown_t = length(tags)

    samples = Sample[]

    lines = open(readlines, path, "r")
    push!(lines, "")

    wordids, charids, tagids, dims_c = Int[], Int[], Int[], Int[]
    for line in lines
        if isempty(line)
            isempty(wordids) && continue

            w = reshape(wordids, 1, length(wordids))
            c = reshape(charids, 1, length(charids))
            dims_w = fill(length(wordids), 1)

            push!(samples, Sample(w, dims_w, c, dims_c, tagids))

            wordids, charids, tagids, dims_c = Int[], Int[], Int[], Int[]
        else
            items = Vector{String}(split(line,"\t"))
            word = strip(items[1])
            push!(wordids, get(word_dic, word, unknown_w))

            chars = Vector{Char}(word)
            append!(charids, map(c -> get(char_dic, c, unknown_c), chars))
            push!(dims_c, length(chars))

            if length(items) >= 2
                tag = strip(items[2])
                push!(tagids, get(tag_dic, tag, unknown_t))
            end
        end
    end

    samples
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
    shuffle && shuffle!(samples)
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
