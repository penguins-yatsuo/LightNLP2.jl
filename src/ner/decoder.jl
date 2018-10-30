using HDF5, ProgressMeter

using Printf: @printf, @sprintf

using Merlin: oncpu, setcpu, ongpu, setcuda, settrain
using Merlin: shuffle!, gradient!, SGD

mutable struct Decoder
    worddict::Dict
    chardict::Dict
    tagdict::Dict
    nn
end


function Decoder(config::Dict, iolog)
    chardict, charembeds, tagdict = initvocab(config["train_file"])
    worddict, wordembeds = wordvec_read(config["wordvec_file"])

    traindata = readdata(config["train_file"], worddict, chardict, tagdict)
    testdata = readdata(config["test_file"], worddict, chardict, tagdict)
    procname = @sprintf("NER[%s]", get!(config, "jobid", "-"))

    nepochs = get!(config, "nepochs", 1)
    batchsize = get!(config, "batchsize", 1)
    n_train = get!(config, "ntrain", length(traindata))
    n_test = get!(config, "ntest", length(testdata))
    use_gpu = get!(config, "use_gpu", false)

    if n_train == nothing || n_train < 1
        n_train = length(traindata)
    end
    if n_test == nothing || n_test < 1
        n_test = length(testdata)
    end

    @printf(iolog, "%s %s train - traindata:%d testdata:%d words:%d, chars:%d tags:%d\n", @timestr, procname, 
            length(traindata), length(testdata), length(worddict), length(chardict), length(tagdict))
    @printf(iolog, "%s %s nepochs:%d batchsize:%d n_train:%d n_test:%d\n", @timestr, procname, 
            nepochs, batchsize, n_train, n_test)

    nn, nntext = create_model(config, length(tagdict))

    @printf(iolog, "%s %s model - %s\n", @timestr, procname, nntext)
    flush(iolog)

    if use_gpu 
        setcuda(0)
        todevice!(nn)
    end

    opt = SGD()
    test_batches = create_batch(testdata, batchsize, n_test)

    for epoch = 1:nepochs
        # train
        settrain(true)
        @printf(stdout, "Epoch: %d\n", epoch)
        @printf(iolog, "%s %s begin epoch %d\n", @timestr, procname, epoch)
        flush(iolog)
 
        shuffle!(traindata)
        batches = create_batch(traindata, batchsize, n_train)
        opt.rate = config["learning_rate"] * batchsize / sqrt(batchsize) / (1 + 0.05*(epoch-1))

        prog = Progress(length(batches))
        loss = 0.0
        for i in 1:length(batches)
            s = batches[i]
            z = nn(Float32, charembeds, wordembeds, s)
            params = gradient!(z)
            for n in 1:length(params)
                if params[n].grad != nothing
                    opt(params[n])
                end
            end

            loss += sum(Array(z.data)) / batchsize
            next!(prog)
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
            z = nn(Float32, charembeds, wordembeds, s)
            append!(preds, z)
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


function decode(dec::Decoder, config::Dict)
    charembeds = Normal(0, 0.01)(Float32, 20, length(dec.chardict))
    worddict, wordembeds = wordvec_read(config["wordvec_file"])

    testdata = readdata(config["test_file"], worddict, dec.chardict, dec.tagdict)
    
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

function create_model(config::Dict, ntags)
    nlayers = get!(config, "nlayers", 1) 
    droprate = get!(config, "droprate", 0.2)

    neural_network = lowercase(get!(config, "neural_network", "UNKNOWN"))

    if neural_network == "conv" 
        winsize_c = get!(config, "winsize_c", 2)
        winsize_w = get!(config, "winsize_w", 5)

        nn = ConvNet(ntags,
            nlayers=nlayers, winsize_c=winsize_c, winsize_w=winsize_w, droprate=droprate)

        text = @sprintf("%s (nlayers:%d winsize_c:%d winsize_w:%d droprate:%f)\n",  
            neural_network, nlayers, winsize_c, winsize_w, droprate)

    elseif neural_network == "lstm"
        winsize_c = get!(config, "winsize_c", 2)
        bidirectional = get!(config, "bidirectional", true)

        nn = LstmNet(ntags,
            nlayers=nlayers, winsize_c=winsize_c, bidirectional=bidirectional, droprate=droprate)

        text = @sprintf("%s (nlayers:%d winsize_c:%d bidirectional:%s droprate:%f)\n", 
            neural_network, nlayers, winsize_c, string(bidirectional), droprate)

    else
        throw(UndefVarError(:neural_network))
    end

    nn, text
end


function fscore(golds::Vector{T}, preds::Vector{T}) where T
    set = intersect(Set(golds), Set(preds))
    count = length(set)
    prec = round(count/length(preds); digits=5)
    recall = round(count/length(golds); digits=5)
    fval = round(2*recall*prec/(recall+prec); digits=5)
    prec, recall, fval
end