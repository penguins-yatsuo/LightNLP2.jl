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

function load_samples(path::String, embeds::Embeds, tagdict::Dict{String, Int}=Dict{String, Int}())
    load_samples(path, embeds.worddict, embeds.unknown_word, embeds.chardict, embeds.unknown_char, tagdict)
end

function load_samples(path::String, worddict::Dict, unknown_word::String, chardict::Dict, unknown_char::Char,
        tagdict::Dict{String, Int}=Dict{String, Int}(), unknown_tag::String="O")

    samples = Sample[]

    unknown_wordid = worddict[unknown_word]
    unknown_charid = chardict[unknown_char]
    unknown_tagid = tagdict[unknown_tag]

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
            push!(wordids, get(worddict, word, unknown_wordid))

            chars = Vector{Char}(word)
            append!(charids, map(c -> get(chardict, c, unknown_charid), chars))
            push!(dims_c, length(chars))

            if length(items) >= 2
                tag = strip(items[2])
                push!(tagids, get(tagdict, tag, unknown_tagid))
            end
        end
    end

    samples
end


mutable struct SampleIterater
    samples
    batchsize
    n_samples
    channel
end

function SampleIterater(samples, batchsize::Int, n_samples::Int, shuffle::Bool)
    (n_samples < 1 || n_samples > length(samples)) && (n_samples = length(samples))
    shuffle && shuffle!(samples)
    SampleIterater(samples, batchsize, n_samples, nothing)
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

    sort!(ss, rev=true, by=(s -> sum(s.dims_w)))

    w = cat(dims=2, map(x -> x.w, ss)...)
    c = cat(dims=2, map(x -> x.c, ss)...)
    dims_w = cat(dims=1, map(x -> x.dims_w, ss)...)
    dims_c = cat(dims=1, map(x -> x.dims_c, ss)...)
    t = (ss[1].t == nothing) ? nothing : cat(dims=1, map(x -> x.t, ss)...)

    Sample(w, dims_w, c, dims_c, t)
end
