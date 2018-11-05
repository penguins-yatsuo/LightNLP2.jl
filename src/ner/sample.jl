struct Sample
    w::Matrix{Int}
    dims_w::Vector{Int}
    c::Matrix{Int}
    dims_c::Vector{Int}
    t::Vector{Int}
end

Base.length(x::Sample) = length(x.dims_w)

function Base.string(x::Sample)
    string("Sample",
        " w=", string(size(x.w)), string(x.w),
        " dims_w=", string(size(x.dims_w)), string(x.dims_w),
        " c=", string(size(x.c)), string(x.c),
        " dims_c=", string(size(x.dims_c)), string(x.dims_c),
        " t=", string(size(x.t)), string(x.t))
end

function load_samples(path::String, embeds::Embeds, initial_tagdict::Dict{String, Int}=Dict{String, Int}())
    load_samples(path, embeds.worddict, embeds.unknown_word, embeds.chardict, embeds.unknown_char, initial_tagdict)
end

function load_samples(path::String, worddict::Dict, unknown_word::String, chardict::Dict, unknown_char::Char,
        initial_tagdict::Dict{String, Int}=Dict{String, Int}())

    samples = Sample[]

    unknown_wordid = worddict[unknown_word]
    unknown_charid = chardict[unknown_char]
    tagdict = copy(initial_tagdict)

    lines = open(readlines, path, "r")
    push!(lines, "")

    wordids, charids, tagids, dims_c = Int[], Int[], Int[], Int[]
    for i = 1:length(lines)
        line = lines[i]

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
                tagid = haskey(tagdict, tag) ? tagdict[tag] : (tagdict[tag] = length(tagdict) + 1)
                push!(tagids, tagid)
            end
        end
    end

    samples, tagdict
end


mutable struct SampleIterater
    samples
    batchsize
    n_samples
    buffer
end

function SampleIterater(samples, batchsize::Int, n_samples::Int, shuffle::Bool)
    (n_samples < 1 || n_samples > length(samples)) && (n_samples = length(samples))
    shuffle && shuffle!(samples)
    SampleIterater(samples, batchsize, n_samples, nothing)
end

function init(iter::SampleIterater)
    iter.buffer = Channel{Sample}(8)
    offsets = range(1, step=iter.batchsize, length=length(iter))
    task = @async foreach(i -> put!(iter.buffer, batch_samples(iter, i)), offsets)
    bind(iter.buffer, task)

    # return first element
    take!(iter.buffer)
end

function Base.iterate(iter::SampleIterater, (element, i)=(init(iter), 1))
    if !isopen(iter.buffer)
        nothing
    else
        i_next = i + iter.batchsize
        element, (take!(iter.buffer), i_next)
    end
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
