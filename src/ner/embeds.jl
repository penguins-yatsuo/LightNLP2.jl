import Merlin: Normal
import HDF5: h5read

import Statistics: mean

UNKNOWN_CHAR = Char(0xfffd) # Unicode U+fffd 'REPLACEMENT CHARACTER'
UNKNOWN_WORD = string("-UNK-")

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
