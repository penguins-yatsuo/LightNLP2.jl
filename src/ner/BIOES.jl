module BIOES

function span_decode(ids::Vector{Int}, tagdict::Dict{String, Int})
    tags = Dict(i => tag for (tag, i) in pairs(tagdict))
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
