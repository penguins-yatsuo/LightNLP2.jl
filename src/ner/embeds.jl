import Merlin: Normal

struct Embeds
    words
    worddict
    word_embeds
    unknown_word
    chars
    chardict
    char_embeds
    unknown_char
end

UNKNOWN_WORD = "UNKNOWN"
UNKNOWN_CHAR = Char(0) #NUL

function Embeds(embeds_file::String; charvec_dim::Int=20)
    words = h5read(embeds_file, "words")
    worddict = Dict(words[i] => i for i=1:length(words))
    word_embeds = h5read(embeds_file, "vectors")

    chars = sort(collect(Set(Iterators.flatten(words))))
    push!(chars, UNKNOWN_CHAR)

    chardict = Dict(chars[i] => i for i=1:length(chars))
    char_embeds = Merlin.Normal(0, 0.01)(Float32, charvec_dim, length(chardict))

    Embeds(words, worddict, word_embeds, UNKNOWN_WORD, chars, chardict, char_embeds, UNKNOWN_CHAR)
end

function Base.string(embeds::Embeds)
    string("words:", length(embeds.words), " chars:", length(embeds.chars))
end
