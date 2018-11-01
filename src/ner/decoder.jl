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
    procname = @sprintf("NER[%s]", get!(args, "jobid", "-"))

    nepochs = getarg!(args, "nepochs", 1)
    batchsize = getarg!(args, "batchsize", 1)
    n_train = getarg!(args, "ntrain", 0)
    n_test = getarg!(args, "ntest", 0)
    use_gpu = getarg!(args, "use_gpu", false)

    embeds = Embeds(args["wordvec_file"], charvec_dim=20)
    train_samples, tagdict = load_samples(args["train_file"], embeds)
    test_samples, tagdict = load_samples(args["test_file"], embeds, tagdict)

    nn = begin
        lowercase(args["neural_network"]) == "conv" ? ConvNet(args) :
        lowercase(args["neural_network"]) == "lstm" ? LstmNet(args) : nothing
    end
    opt = SGD()

    @printf(iolog, "%s %s model - %s\n", @timestr, procname, string(nn))
    @printf(iolog, "%s %s embeds - %s\n", @timestr, procname, string(embeds))
    @printf(iolog, "%s %s data - traindata:%d testdata:%d tags:%d\n", @timestr, procname,
            length(train_samples), length(test_samples), length(tagdict))
    @printf(iolog, "%s %s train - nepochs:%d batchsize:%d n_train:%d n_test:%d\n", @timestr, procname,
            nepochs, batchsize, n_train, n_test)
    flush(iolog)

    if use_gpu
        setcuda(0)
        todevice!(nn)
    end

    test_iter = SampleIterater(test_samples, batchsize, n_test, false)

    for epoch = 1:nepochs
        # train
        settrain(true)
        @printf(stdout, "Epoch: %d\n", epoch)
        @printf(iolog, "%s %s begin epoch %d\n", @timestr, procname, epoch)
        flush(iolog)

        train_iter = SampleIterater(train_samples, batchsize, n_test, false)

        prog = ProgressMeter.Progress(length(train_iter))
        opt.rate = args["learning_rate"] * batchsize / sqrt(batchsize) / (1 + 0.05*(epoch-1))

        loss = 0.0
        for (i, s) in enumerate(train_iter)
            z = nn(Float32, embeds.char_embeds, embeds.word_embeds, s)
            params = gradient!(z)
            foreach(opt, filter(x -> !isnothing(x.grad), params))
            loss += sum(Array(z.data)) / length(s.dims_w)
            ProgressMeter.next!(prog)
        end
        loss /= length(train_iter)
        @printf(stdout, "Loss: %.5f\n", loss)

        # test
        settrain(false)
        @printf(stdout, "Test ")
        preds = Int[]
        golds = Int[]
        for (i, s) in enumerate(test_iter)
            pred = nn(Float32, embeds.char_embeds, embeds.word_embeds, s)
            append!(preds, pred)
            append!(golds, s.t)
        end
        length(preds) == length(golds) || throw("Length mismatch: $(length(preds)), $(length(golds))")

        preds = BIOES.span_decode(preds, tagdict)
        golds = BIOES.span_decode(golds, tagdict)
        prec, recall, fval = BIOES.fscore(golds, preds)

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

    Decoder(embeds.worddict, embeds.chardict, tagdict, nn)
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
