import ProgressMeter
import Serialization

using Printf: @printf, @sprintf
using Merlin: oncpu, setcpu, ongpu, setcuda, settrain, isnothing
using Merlin: shuffle!, gradient!, SGD

mutable struct Decoder
    tagdict
    chardict
    char_embeds
    net
end

function Decoder(tags::Array{String}=["B", "I", "O", "E", "S"])
    tagdict = Dict(tags[i] => i for i=1:length(tags))
    chardict = Dict{Char, Int}()
    Decoder(tagdict, chardict, nothing, nothing)
end

function Decoder(fname::String)
    load(fname)
end

function save(model::Decoder, fname::String)
    open(io -> Serialization.serialize(io, model), fname, "w")
end

function load(fname::String)
    open(Serialization.deserialize, fname)
end

function train!(model::Decoder, args::Dict, iolog=stdout)
    procname = @sprintf("NER[%s]", get!(args, "jobid", "-"))

    # read word embeds
    embeds = Embeds(args["wordvec_file"]; charvec_dim=20)
    model.chardict = embeds.chardict
    model.char_embeds = embeds.char_embeds

    # read samples
    train_samples = load_samples(args["train_file"], embeds, model.tagdict)
    test_samples = load_samples(args["test_file"], embeds, model.tagdict)

    # get args
    use_gpu = getarg!(args, "use_gpu", 0)
    nepochs = getarg!(args, "nepochs", 1)
    batchsize = getarg!(args, "batchsize", 1)
    ntags = getarg!(args, "ntags", length(model.tagdict))
    n_train = getarg!(args, "ntrain", length(train_samples))
    n_test = getarg!(args, "ntest", length(test_samples))

    # create neural network
    model.net = begin
        lowercase(args["neural_network"]) == "conv" ? ConvNet(args) :
        lowercase(args["neural_network"]) == "lstm" ? LstmNet(args) : nothing
    end

    # optimizer
    opt = SGD()

    # print job status
    @printf(iolog, "%s %s model - %s\n", @timestr, procname, string(model.net))
    @printf(iolog, "%s %s embeds - %s\n", @timestr, procname, string(embeds))
    @printf(iolog, "%s %s data - traindata:%d testdata:%d\n", @timestr, procname,
            length(train_samples), length(test_samples))
    @printf(iolog, "%s %s train - nepochs:%d batchsize:%d n_train:%d n_test:%d use_gpu:%s\n",
            @timestr, procname, nepochs, batchsize, n_train, n_test, string(use_gpu))
    flush(iolog)

    # GPU device setup
    if use_gpu
        setcuda(0)
        todevice!(model.net)
    end

    # test data iterator
    test_iter = SampleIterater(test_samples, batchsize, n_test, false)

    # start train
    for epoch = 1:nepochs
        @printf(stdout, "Epoch: %d\n", epoch)
        @printf(iolog, "%s %s begin epoch %d\n", @timestr, procname, epoch)
        flush(iolog)

        # train
        settrain(true)
        train_iter = SampleIterater(train_samples, batchsize, n_train, true)

        prog = ProgressMeter.Progress(length(train_iter), desc="Train: ")
        opt.rate = args["learning_rate"] * batchsize / sqrt(batchsize) / (1 + 0.05*(epoch-1))

        loss = 0.0
        for (i, s) in enumerate(train_iter)
            z = model.net(Float32, embeds.char_embeds, embeds.word_embeds, s)
            params = gradient!(z)
            foreach(opt, filter(x -> !isnothing(x.grad), params))
            loss += sum(z.data) / length(s)
            ProgressMeter.next!(prog)
        end
        loss /= length(train_iter)
        ProgressMeter.finish!(prog, showvalues=["Loss" => round(loss, digits=5)] )

        # test
        @printf(stdout, "Test: ")
        settrain(false)
        preds = Int[]
        golds = Int[]
        for (i, s) in enumerate(test_iter)
            pred, prob = model.net(Float32, embeds.char_embeds, embeds.word_embeds, s)
            append!(preds, pred)
            append!(golds, s.t)
        end

        # evaluation of this epochs
        @assert length(preds) == length(golds)
        span_preds = BIOES.span_decode(preds, model.tagdict)
        span_golds = BIOES.span_decode(golds, model.tagdict)
        prec, recall, fval = BIOES.fscore(span_golds, span_preds)
        println(span_preds)

        @printf(stdout, "Prec: %.5f, Recall: %.5f, Fscore: %.5f\n", prec, recall, fval)
        @printf(iolog, "%s %s end epoch %d - loss:%.5f fval:%.5f prec:%.5f recall:%.5f\n", @timestr, procname,
                epoch, loss, fval, prec, recall)
        flush(iolog)
    end

    @printf(iolog, "%s %s training complete\n", @timestr, procname)

    # fetch model from GPU device
    if !oncpu()
        setcpu()
        todevice!(model.net)
    end
end


function decode(model::Decoder, args::Dict, iolog=stdout)

    # read word embeds
    embeds = Embeds(args["wordvec_file"]; charvec_dim=20)
    embeds.chars = sort(collect(keys(model.chardict)))
    embeds.chardict = model.chardict
    embeds.char_embeds = model.char_embeds

    # read samples
    samples = load_samples(args["test_file"], embeds, model.tagdict)

    # get args
    use_gpu = getarg!(args, "use_gpu", 0)
    batchsize = getarg!(args, "batchsize", 1)
    n_pred = length(samples)

    # GPU device setup
    if use_gpu
        setcuda(0)
        todevice!(model.net)
    end

    # iterator
    pred_iter = SampleIterater(samples, batchsize, n_pred, false)

    settrain(false)
    preds = Int[]
    probs = nothing
    for (i, s) in enumerate(pred_iter)
        pred, prob = model.net(Float32, embeds.char_embeds, embeds.word_embeds, s)
        append!(preds, pred)
        probs = (probs == nothing) ? prob : cat(dims=2, probs, prob)
    end
    tag_preds = BIOES.decode(preds, model.tagdict)

    lines = open(readlines, args["test_file"])
    i = 1
    for line in lines
        if !isempty(strip(line))
            tag = tag_preds[i]
            prob = join(map(p -> round(p, digits=3), view(probs, :, i)), "\t")
            println("$line\t$tag\t$prob")
            i += 1
        else
            println("")
        end
    end
end

function getarg!(args::Dict, key::String, default::Any)
    (get!(args, key, default) == nothing) ? args[key] = default : args[key]
end
