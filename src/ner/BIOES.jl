module BIOES


function span_decode(ids::Vector{Int}, tags::Vector{String})
    spans = Tuple{Int, Int, String}[]
    bpos = 0
    for i = 1:length(ids)
        tag = tags[ids[i]]
        tag == "O" && continue

        startswith(tag, "B") && (bpos = i)
        startswith(tag, "S") && (bpos = i)

        if startswith(tag, "S") || (startswith(tag, "E") && (bpos > 0))
            tag = tags[ids[bpos]]
            basetag = (length(tag) > 2) ? tag[3:end] : ""
            push!(spans, (bpos, i, basetag))
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
            prob = join(map(p -> round(p, digits=3), view(probs, :, i)), "\t")
            push!(merged, "$line\t$tag\t$prob")
            println("$i $line\t$tag\t$prob")
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

end
