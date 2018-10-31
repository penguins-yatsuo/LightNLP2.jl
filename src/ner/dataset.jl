import Merlin: Normal

struct Sample
    w::Matrix{Int}
    dims_w::Vector{Int}
    c::Matrix{Int}
    dims_c::Vector{Int}
    t::Vector{Int}
end

struct Dataset
    data_length
    words
    worddict
    word_embeds
    chars
    chardict    
    char_embeds
    tags
    tagdict
    samples::Vector{Sample}
end

UNKNOWN_WORD = "UNKNOWN"
UNKNOWN_CHAR = Char(0)

function Dataset(embeds_file::String, data_file::String, batchsize::Int=0, n_samples::Int=0)
    words = h5read(embeds_file, "words")
    word_embeds = h5read(embeds_file, "vectors")
    chars = sort(collect(Set(Iterators.flatten(words))))
    Dataset(words, word_embeds, chars, data_file, batchsize, n_samples)
end

function Dataset(words::Vector{String}, word_embeds::Matrix{T}, chars::Vector{Char}, data_file::String, batchsize::Int=0, n_samples::Int=0) where T
    worddict = Dict(words[i] => i for i=1:length(words))
    # worddict[UNKNOWN_WORD] = length(worddict) + 1

    chardict = Dict(chars[i] => i for i=1:length(chars))
    chardict[UNKNOWN_CHAR] = length(chardict) + 1

    char_embeds = Merlin.Normal(0, 0.01)(Float32, 20, length(chardict))
    samples, tagdict = load_bioes(data_file, worddict, chardict)
    
    data_length = length(samples)
    (n_samples > 0) && (samples = view(samples, 1:n_samples))

    if batchsize > 0
        Dataset(data_length, words, worddict, word_embeds, chars, chardict, char_embeds, keys(tagdict), tagdict, batch(samples, batchsize))
    else
        Dataset(data_length, words, worddict, word_embeds, chars, chardict, char_embeds, keys(tagdict), tagdict, samples)
    end
end

function Base.length(x::Dataset)
    x.data_length
end

function Base.string(x::Sample)   
    string("Sample", 
        " w=", string(size(x.w)), string(x.w), 
        " dims_w=", string(size(x.dims_w)), string(x.dims_w), 
        " c=", string(size(x.c)), string(x.c), 
        " dims_c=", string(size(x.dims_c)), string(x.dims_c), 
        " t=", string(size(x.t)), string(x.t))
end


function load_bioes(path::String, worddict::Dict, chardict::Dict)
    samples = Sample[]
    tagdict = Dict{String,Int64}()

    lines = open(readlines, path)
    push!(lines, "")

    wordids, charids, tagids, dims_c = Int[], Int[], Int[], Int[], Int[]
    for i = 1:length(lines)
        line = lines[i]

        if isempty(line)
            isempty(wordids) && continue
            
            w = reshape(wordids, 1, length(wordids))
            c = reshape(charids, 1, length(charids))
            dims_w = fill(length(wordids), 1)

            push!(samples, Sample(w, dims_w, c, dims_c, tagids))  

            wordids, charids, tagids, dims_c = Int[], Int[], Int[], Int[], Int[]
        else            
            items = Vector{String}(split(line,"\t"))
            word = strip(items[1])
            push!(wordids, get(worddict, lowercase(word), worddict[UNKNOWN_WORD]))
            
            chars = Vector{Char}(word)
            append!(charids, map(c -> get(chardict, c, chardict[UNKNOWN_CHAR]), chars))
            push!(dims_c, length(chars))

            if length(items) >= 2
                tag = strip(items[2])
                push!(tagids, get!(tagdict, tag, length(tagdict) + 1))
            end
        end
    end

    samples, tagdict
end


function batch(samples::Vector{Sample}, batchsize::Int, n_samples::Int=0)
    batch_samples = Sample[]

    n_samples = (n_samples < 1 || n_samples > length(samples)) ? length(samples) : n_samples

    for i = 1:batchsize:n_samples
        s = samples[i:min(i + batchsize - 1, n_samples)]

        w = cat(dims=2, map(x -> x.w, s)...)
        c = cat(dims=2, map(x -> x.c, s)...)
        batchdims_w = cat(dims=1, map(x -> x.dims_w, s)...)
        batchdims_c = cat(dims=1, map(x -> x.dims_c, s)...)
        t = (s[1].t == nothing) ? nothing : cat(dims=1, map(x -> x.t, s)...)

        push!(batch_samples, Sample(w, batchdims_w, c, batchdims_c, t))
    end
    batch_samples
end
