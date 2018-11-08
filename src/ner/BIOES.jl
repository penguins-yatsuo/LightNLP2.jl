import Formatting

function load_samples(path::String, words::Vector{String}, chars::Vector{Char}, tags::Vector{String})

    word_dic = Dict(words[i] => i for i in 1:length(words))
    char_dic = Dict(chars[i] => i for i in 1:length(chars))
    tag_dic = Dict(tags[i] => i for i in 1:length(tags))

    unknown_w = length(words)
    unknown_c = length(chars)
    unknown_t = tag_dic["O"]

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


function span_decode(ids::Vector{Int}, tags::Vector{String})
    spans = Tuple{Int, Int, String}[]
    bpos = 0
    for pos in 1:length(ids)
        tag = tags[ids[pos]]
        tag == "O" && continue

        startswith(tag, "B") && (bpos = pos)
        startswith(tag, "S") && (bpos = pos)

        if startswith(tag, "S") || (startswith(tag, "E") && (bpos > 0))
            btag = tags[ids[bpos]]
            suffix = (length(btag) > 2) ? btag[3:end] : ""
            push!(spans, (bpos, pos, suffix))
            bpos = 0
        end
    end
    spans
end

function merge_decode(lines::Vector{String}, tags::Vector{String}, preds::Matrix{Int}, probs::Matrix{Float32})
    merged = String[]
    i = 1
    for line in lines
        if !isempty(strip(line))
            tag = join(map(t -> tags[t], view(preds, :, i)), "\t")
            prob = join(map(p -> Formatting.format("{1:.2f}", p), view(probs, :, i)), "\t")
            push!(merged, "$line\t$tag\t$prob")
            i += 1
        else
            push!(merged, line)
        end
    end
    merged
end

function fscore(golds::Vector{T}, preds::Vector{T}) where T
    match = intersect(Set(golds), Set(preds))
    count = length(match)
    if count == 0
        prec, recall, fval = 0, 0, 0
    else
        prec = round(count / length(preds); digits=5)
        recall = round(count / length(golds); digits=5)
        fval = round(2 * recall * prec / (recall + prec); digits=5)
    end
    prec, recall, fval
end
