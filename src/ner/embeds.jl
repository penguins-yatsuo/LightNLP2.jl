import Merlin: Normal
import HDF5: h5read

import Statistics: mean

UNKNOWN_CHAR = Char(0xfffd) # Unicode U+fffd 'REPLACEMENT CHARACTER'
UNKNOWN_WORD = string(UNKNOWN_CHAR)

function embed_words(embeds_file::String)
    h5read(embeds_file, "words")
end

function embed_wordvecs!(words::Vector{String}, embeds_file::String)
    wordvecs = h5read(embeds_file, "vectors")

    if last(words) != UNKNOWN_WORD
        push!(words, UNKNOWN_WORD)
        wordvecs = hcat(wordvecs, zeros(eltype(wordvecs), size(wordvecs, 1)))    
    end

    words, wordvecs
end

function embed_chars(words::Vector{String})
    sort(collect(Set(Iterators.flatten(words))))
end

function embed_charvecs!(chars::Vector{Char}; csize::Int=20)
    charvecs = Merlin.Normal(0, 0.01)(Float32, csize, length(chars))

    if last(chars) != UNKNOWN_CHAR
        push!(chars, UNKNOWN_CHAR)
        charvecs = hcat(charvecs, zeros(eltype(charvecs), size(charvecs, 1)))
    end

    chars, charvecs
end

function read_wordvecs(embeds_file::String)
    words = h5read(embeds_file, "words")
    wordvecs = h5read(embeds_file, "vectors")

    if last(words) != UNKNOWN_WORD
        push!(words, UNKNOWN_WORD)
        wordvecs = hcat(wordvecs, mean(wordvecs, dims=2))    
    end

    words, wordvecs
end

function create_charvecs(words::Vector{String}; csize::Int=20)
    chars = sort(collect(Set(Iterators.flatten(words))))
    charvecs = Merlin.Normal(0, 0.01)(Float32, csize, length(chars))

    if last(chars) != UNKNOWN_CHAR
        push!(chars, UNKNOWN_CHAR)
        charvecs = hcat(charvecs, mean(charvecs, dims=2))
    end

    chars, charvecs
end
