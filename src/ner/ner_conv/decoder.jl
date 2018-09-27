export decode

using Printf, Dates, HDF5, Random, Merlin, ProgressMeter

mutable struct Decoder
    worddict::Dict
    chardict::Dict
    tagdict::Dict
    nn
end

struct Sample
    w
    batchdims_w
    c
    batchdims_c
    t
end

function Base.string(x::Sample)   
    string("Sample", 
        " w=", string(size(x.w)), string(x.w), 
        " dim_w=", string(size(x.batchdims_w)), string(x.batchdims_w), 
        " c=", string(size(x.c)), string(x.c), 
        " dim_c=", string(size(x.batchdims_c)), string(x.batchdims_c), 
        " t=", string(size(x.t)), string(x.t))
end



macro logtime()
    return :( Dates.format(now(), "yyyy-mm-dd HH:MM:SS") )
end

function create_batch(samples::Vector{Sample}, batchsize::Int)
    batches = Sample[]
    for i = 1:batchsize:length(samples)
        range = i:min(i+batchsize-1,length(samples))
        s = samples[range]
        w = cat(dims=2, map(x -> x.w, s)...)
        c = cat(dims=2, map(x -> x.c, s)...)
        batchdims_w = cat(dims=1, map(x -> x.batchdims_w, s)...)
        batchdims_c = cat(dims=1, map(x -> x.batchdims_c, s)...)
        t = s[1].t == nothing ? nothing : cat(dims=1, map(x -> x.t, s)...)

        w = w
        batchdims_w = sort(batchdims_w, rev=true)
        c = c
        batchdims_c = batchdims_c
        t = t

        push!(batches, Sample(w, batchdims_w, c, batchdims_c, t))
    end
    batches
end

function Decoder(config::Dict, iolog)
    words = h5read(config["wordvec_file"], "words")
    wordembeds = h5read(config["wordvec_file"], "vectors")
    worddict = Dict(words[i] => i for i=1:length(words))
    chardict, tagdict = initvocab(config["train_file"])
    charembeds = Normal(0,0.01)(Float32, 20, length(chardict))
    traindata = readdata(config["train_file"], worddict, chardict, tagdict)
    testdata = readdata(config["test_file"], worddict, chardict, tagdict)
    nepochs = haskey(config, "nepochs") ? config["nepochs"] : 1
    nlayers = haskey(config, "nlayers") ? config["nlayers"] : 1
    winsize_c = haskey(config, "winsize_c") ? config["winsize_c"] : 2
    winsize_w = haskey(config, "winsize_w") ? config["winsize_w"] : 5  
    droprate = haskey(config, "droprate") ? config["droprate"] : 0.2 

    procname = @sprintf("NER.Convolution[%s]", haskey(config, "jobid") ? config["jobid"] : "-")
    # @printf(iolog, "%s %s training - train:%d test:%d words:%d, chars:%d tags:%d\n", @logtime, procname, 
    #         length(traindata), length(testdata), length(worddict), length(chardict), length(tagdict))
    # @printf(iolog, "%s %s arguments - nepochs:%d, nlayers:%d droprate:%f winsize_c:%d winsize_w:%d\n", @logtime, procname, 
    #         nepochs, nlayers, droprate, winsize_c, winsize_w)
    flush(iolog)

    nn = NN(wordembeds, charembeds, length(tagdict); 
            nlayers=nlayers, winsize_c=winsize_c, winsize_w=winsize_w, droprate=droprate)

    # @info("#Training examples:\t$(length(traindata))")
    # @info("#Testing examples:\t$(length(testdata))")
    # @info("#Words1:\t$(length(worddict))")
    # @info("#Chars:\t$(length(chardict))")
    # @info("#Tags:\t$(length(tagdict))")

    batchsize = config["batchsize"]
    test_batches = create_batch(testdata, batchsize)

    opt = SGD()
    for epoch = 1:nepochs
        println("Epoch:\t$epoch")
 
        #opt.rate = LEARN_RATE / BATCHSIZE
        # opt.rate = config["learning_rate"] * batchsize / sqrt(batchsize) / (1 + 0.05*(epoch-1))
        # println("Learning rate: $(opt.rate)")

        @printf(iolog, "%s %s begin epoch %d - optimize %s\n", @logtime, procname, epoch, string(opt))
        flush(iolog)

        shuffle!(traindata)
        batches = create_batch(traindata, batchsize)
        prog = Progress(length(batches))
        loss = 0.0
        for i in 1:length(batches)
            s = batches[i]

            z = nn(Float32, s, true)
            params = gradient!(z)
            for n in 1:length(params)
                if !isnothing(params[n].grad)
                    opt(params[n])
                end
            end

            loss += sum(Array(z.data))
            ProgressMeter.next!(prog)
        end
        loss /= length(batches)
        println("Loss:\t$loss")

        # test
        println("Testing...")
        preds = Int[]
        golds = Int[]
        for i in 1:length(test_batches)
            s = test_batches[i]
            z = nn(Float32, s, false)
            append!(preds, z)
            append!(golds, s.t)
        end
        length(preds) == length(golds) || throw("Length mismatch: $(length(preds)), $(length(golds))")

        preds = BIOES.decode(preds, tagdict)
        golds = BIOES.decode(golds, tagdict)
        prec, recall, fval = fscore(golds, preds)

        @printf(iolog, "%s %s end epoch %d - loss:%.4e fval:%.5f prec:%.5f recall:%.5f\n", @logtime, procname, 
                epoch, loss, fval, prec, recall)
        flush(iolog)
        println()
    end

    @printf(iolog, "%s %s training complete\n", @logtime, procname)

    Decoder(worddict, chardict, tagdict, nn)
end

function initvocab(path::String)
    chardict = Dict{String,Int}()
    tagdict = Dict{String,Int}()
    lines = open(readlines, path)
    for line in lines
        isempty(line) && continue
        items = Vector{String}(split(line,"\t"))
        word = strip(items[1])
        chars = Vector{Char}(word)
        for c in chars
            c = string(c)
            if haskey(chardict, c)
                chardict[c] += 1
            else
                chardict[c] = 1
            end
        end

        tag = strip(items[2])
        haskey(tagdict,tag) || (tagdict[tag] = length(tagdict)+1)
    end

    chars = String[]
    for (k,v) in chardict
        v >= 3 && push!(chars,k)
    end
    chardict = Dict(chars[i] => i for i=1:length(chars))
    chardict["UNKNOWN"] = length(chardict) + 1

    chardict, tagdict
end

function decode(dec::Decoder, config::Dict)
    testdata = readdata(config["test_file"], dec.worddict, dec.chardict, dec.tagdict)
    testdata = create_batch(testdata, 10)
    id2tag = Array{String}(length(dec.tagdict))
    for (k, v) in dec.tagdict
        id2tag[v] = k
    end

    preds = Int[]
    for x in testdata
        y = dec.nn(x, false)
        append!(preds, y)
    end

    lines = open(readlines, config["test_file"])
    i = 1
    for line in lines
        if !isempty(strip(line))
            tag = id2tag[preds[i]]
            println("$line\t$tag")
            i += 1
        else
            println("")            
        end
    end
end

function readdata(path::String, worddict::Dict, chardict::Dict, tagdict::Dict)
    samples = Sample[]
    words, tags = String[], String[]
    unkword = worddict["UNKNOWN"]
    unkchar = chardict["UNKNOWN"]

    lines = open(readlines, path)
    push!(lines, "")
    for i = 1:length(lines)
        line = lines[i]
        if isempty(line)
            isempty(words) && continue
            wordids = Int[]
            charids = Int[]
            batchdims_c = Int[]
            for w in words
                #w0 = replace(lowercase(w), r"[0-9]", '0')
                id = get(worddict, lowercase(w), unkword)
                push!(wordids, id)

                chars = Vector{Char}(w)
                ids = map(chars) do c
                    get(chardict, string(c), unkchar)
                end
                append!(charids, ids)
                push!(batchdims_c, length(ids))
            end
            batchdims_w = [length(words)]
            w = reshape(wordids, 1, length(wordids))
            c = reshape(charids, 1, length(charids))
            t = isempty(tags) ? nothing : map(t -> tagdict[t], tags)
            push!(samples, Sample(w, batchdims_w, c, batchdims_c, t))
            empty!(words)
            empty!(tags)
        else
            items = Vector{String}(split(line,"\t"))
            word = strip(items[1])
            @assert !isempty(word)
            push!(words, word)
            if length(items) >= 2
                tag = strip(items[2])
                push!(tags, tag)
            end
        end
    end
    samples
end

function fscore(golds::Vector{T}, preds::Vector{T}) where T
    set = intersect(Set(golds), Set(preds))
    count = length(set)
    prec = round(count/length(preds); digits=5)
    recall = round(count/length(golds); digits=5)
    fval = round(2*recall*prec/(recall+prec); digits=5)
    println("Prec:\t$prec")
    println("Recall:\t$recall")
    println("Fscore:\t$fval")
    prec, recall, fval
end
