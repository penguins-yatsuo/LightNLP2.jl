import Merlin: Normal
import HDF5: h5read

import Statistics: mean

UNKNOWN_CHAR = Char(0xfffd) # Unicode U+fffd 'REPLACEMENT CHARACTER'
UNKNOWN_WORD = string(UNKNOWN_CHAR)

function load_embeds(embeds_file::String; csize::Int=20)
    words = h5read(embeds_file, "words")
    wordvecs = h5read(embeds_file, "vectors")

    push!(words, UNKNOWN_WORD)
    wordvecs = hcat(wordvecs, mean(wordvecs, dims=2))

    chars = sort(collect(Set(Iterators.flatten(words))));
    chavecs = Merlin.Normal(0, 0.01)(Float32, csize, length(chars))

    words, wordvecs, chars, chavecs
end
