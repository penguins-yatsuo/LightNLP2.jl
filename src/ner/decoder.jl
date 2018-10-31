using HDF5, ProgressMeter

using Printf: @printf, @sprintf

using Merlin: oncpu, setcpu, ongpu, setcuda, settrain, isnothing
using Merlin: shuffle!, gradient!, SGD

mutable struct Decoder
    worddict::Dict
    chardict::Dict
    tagdict::Dict
    nn
end

function getarg!(args::Dict, key::String, default::Any)
    (get!(args, key, default) == nothing) ? args[key] = default : args[key]
end

function Decoder(args::Dict, iolog)

    nepochs = getarg!(args, "nepochs", 1)
    batchsize = getarg!(args, "batchsize", 1)
    n_train = getarg!(args, "ntrain", 0)
    n_test = getarg!(args, "ntest", 0)
    use_gpu = getarg!(args, "use_gpu", false)


    train_data = Dataset(args["wordvec_file"], args["train_file"], batchsize, n_train)

    chardict = train_data.chardict
    charembeds = train_data.char_embeds

    words = train_data.words
    worddict = train_data.worddict
    wordembeds = train_data.word_embeds
    tagdict = train_data.tagdict

    test_data = Dataset(args["wordvec_file"], args["test_file"], batchsize, n_test)

    # chardict, charembeds, tagdict = initvocab(args["train_file"])
    # words, worddict, wordembeds = wordvec_read(args["wordvec_file"])

    # traindata = readdata(args["train_file"], worddict, chardict, tagdict)
    # testdata = readdata(args["test_file"], worddict, chardict, tagdict)
    procname = @sprintf("NER[%s]", get!(args, "jobid", "-"))


    nn = begin
        lowercase(args["neural_network"]) == "conv" ? ConvNet(args) :
        lowercase(args["neural_network"]) == "lstm" ? LstmNet(args) : nothing
    end

    @printf(iolog, "%s %s model - %s\n", @timestr, procname, string(nn))
    @printf(iolog, "%s %s data - traindata:%d testdata:%d words:%d chars:%d tags:%d\n", @timestr, procname, 
            length(train_data), length(test_data), length(worddict), length(chardict), length(tagdict))
    @printf(iolog, "%s %s train - nepochs:%d batchsize:%d n_train:%d n_test:%d\n", @timestr, procname, 
            nepochs, batchsize, n_train, n_test)
    flush(iolog)

    if use_gpu 
        setcuda(0)
        todevice!(nn)
    end

    opt = SGD()
    test_batches = test_data.samples

    for epoch = 1:nepochs
        # train
        settrain(true)
        @printf(stdout, "Epoch: %d\n", epoch)
        @printf(iolog, "%s %s begin epoch %d\n", @timestr, procname, epoch)
        flush(iolog)
 
        shuffle!(train_data.samples)
        batches = train_data.samples
        opt.rate = args["learning_rate"] * batchsize / sqrt(batchsize) / (1 + 0.05*(epoch-1))

        prog = ProgressMeter.Progress(length(batches))
        loss = 0.0
        for i in 1:length(batches)
            s = batches[i]
            z = nn(Float32, charembeds, wordembeds, s)
            params = gradient!(z)
            foreach(opt, filter(x -> !isnothing(x.grad), params))
            loss += sum(Array(z.data)) / batchsize            
            ProgressMeter.next!(prog)
        end
        loss /= length(batches)
        @printf(stdout, "Loss: %.5f\n", loss)

        # test
        settrain(false)
        @printf(stdout, "Test ")
        preds = Int[]
        golds = Int[]
        for i in 1:length(test_batches)
            s = test_batches[i]
            pred = nn(Float32, charembeds, wordembeds, s)
            append!(preds, pred)
            append!(golds, s.t)
        end
        length(preds) == length(golds) || throw("Length mismatch: $(length(preds)), $(length(golds))")

        preds = BIOES.decode(preds, tagdict)
        golds = BIOES.decode(golds, tagdict)
        prec, recall, fval = fscore(golds, preds)

        @printf(stdout, "Prec: %.5f, Recall: %.5f, Fscore: %.5f\n", prec, recall, fval)
        @printf(iolog, "%s %s end epoch %d - loss:%.5f fval:%.5f prec:%.5f recall:%.5f\n", @timestr, procname, 
                epoch, loss, fval, prec, recall)
        flush(iolog)
        println()
    end

    @printf(iolog, "%s %s training complete\n", @timestr, procname)

    if !oncpu()
        setcpu()
        todevice!(nn)
    end
    Decoder(worddict, chardict, tagdict, nn)
end


function decode(dec::Decoder, args::Dict)
    charembeds = Normal(0, 0.01)(Float32, 20, length(dec.chardict))
    worddict, wordembeds = wordvec_read(args["wordvec_file"])

    testdata = readdata(args["test_file"], worddict, dec.chardict, dec.tagdict)
    
    test_batches = create_batch(testdata, 10)
    id2tag = Array{String}(length(dec.tagdict))
    for (k, v) in dec.tagdict
        id2tag[v] = k
    end

    settrain(false)
    preds = Int[]
    for i in 1:length(test_batches)
        s = test_batches[i]
        y = dec.nn(Float32, charembeds, wordembeds, s)
        append!(preds, y)
    end

    lines = open(readlines, args["test_file"])
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

function fscore(golds::Vector{T}, preds::Vector{T}) where T
    set = intersect(Set(golds), Set(preds))
    count = length(set)
    prec = round(count/length(preds); digits=5)
    recall = round(count/length(golds); digits=5)
    fval = round(2*recall*prec/(recall+prec); digits=5)
    prec, recall, fval
end