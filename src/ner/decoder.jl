export Decoder, save, load, train!, decode


using ProgressMeter

using Printf: @printf, @sprintf
using JLD2: JLDWriteSession, jldopen, read, write

using Merlin: oncpu, setcpu, ongpu, setcuda, settrain, isnothing
using Merlin: shuffle!, gradient!, SGD


mutable struct Decoder
    words::Vector{String}
    wordvecs::Matrix{Float32}
    chars::Vector{Char}
    charvecs::Matrix{Float32}
    tags::Vector{String}
    net
end

function Decoder(words, wordvecs, chars, charvecs, tags)
    Decoder(words, wordvecs, chars, charvecs, tags, nothing)
end

function Decoder(fname::String)
    load(fname)
end

function save(m::Decoder, fname::String)
    jldopen(fname, "w") do file
        wsession = JLDWriteSession()
        write(file, "decoder", m, wsession)
    end
end

function load(fname::String)
    jldopen(fname, "r") do file
        read(file, "decoder")
    end
end

function train!(m::Decoder, args::Dict, iolog=stdout)
    procname = @sprintf("NER[%s]", get!(args, "jobid", "-"))

    # read samples
    train_samples = load_samples(args["train_file"], m.words, m.chars, m.tags)
    test_samples = load_samples(args["test_file"], m.words, m.chars, m.tags)

    # get args
    use_gpu = getarg!(args, "use_gpu", 0)
    nepochs = getarg!(args, "nepochs", 1)
    batchsize = getarg!(args, "batchsize", 1)
    ntags = getarg!(args, "ntags", length(m.tags))
    n_train = getarg!(args, "ntrain", length(train_samples))
    n_test = getarg!(args, "ntest", length(test_samples))

    # create neural network
    m.net = begin
        lowercase(args["neural_network"]) == "conv" ? ConvNet(args) :
        lowercase(args["neural_network"]) == "lstm" ? LstmNet(args) : nothing
    end

    # optimizer
    opt = SGD()

    # print job status
    @printf(iolog, "%s %s model - %s\n", @timestr, procname, string(m.net))
    @printf(iolog, "%s %s embeds - words:%d chars:%d\n", @timestr, procname,
            length(m.wordvecs), length(m.charvecs))
    @printf(iolog, "%s %s data - traindata:%d testdata:%d\n", @timestr, procname,
            length(train_samples), length(test_samples))
    @printf(iolog, "%s %s train - nepochs:%d batchsize:%d n_train:%d n_test:%d use_gpu:%s\n",
            @timestr, procname, nepochs, batchsize, n_train, n_test, string(use_gpu))
    flush(iolog)

    # GPU device setup
    if use_gpu
        setcuda(0)
        todevice!(m.net)
    end

    # test data iterator
    test_iter = SampleIterater(test_samples, batchsize, n_test, shuffle=false, sort=true)

    # start train
    for epoch = 1:nepochs
        @printf(stdout, "Epoch: %d\n", epoch)
        @printf(iolog, "%s %s begin epoch %d\n", @timestr, procname, epoch)
        flush(iolog)

        # train
        settrain(true)
        train_iter = SampleIterater(train_samples, batchsize, n_train, shuffle=true, sort=true)

        progress = ProgressMeter.Progress(length(train_iter), desc="Train: ")
        opt.rate = args["learning_rate"] * batchsize / sqrt(batchsize) / (1 + 0.05*(epoch-1))

        loss = 0.0
        for (i, s) in enumerate(train_iter)
            z = m.net(Float32, m.charvecs, m.wordvecs, s)
            params = gradient!(z)
            foreach(opt, filter(x -> !isnothing(x.grad), params))
            loss += sum(z.data) / length(s)
            ProgressMeter.next!(progress)
        end
        loss /= length(train_iter)
        ProgressMeter.finish!(progress, showvalues=["Loss" => round(loss, digits=5)] )

        # test
        @printf(stdout, "Test: ")
        settrain(false)
        preds = Int[]
        golds = Int[]
        for (i, s) in enumerate(test_iter)
            pred, prob = m.net(Float32, m.charvecs, m.wordvecs, s)
            append!(preds, pred)
            append!(golds, s.t)
        end

        # evaluation of this epochs
        @assert length(preds) == length(golds)
        span_preds = span_decode(preds, m.tags)
        span_golds = span_decode(golds, m.tags)
        prec, recall, fval = fscore(span_golds, span_preds)

        @printf(stdout, "Prec: %.5f, Recall: %.5f, Fscore: %.5f\n", prec, recall, fval)
        @printf(iolog, "%s %s end epoch %d - loss:%.5f fval:%.5f prec:%.5f recall:%.5f\n", @timestr, procname,
                epoch, loss, fval, prec, recall)
        flush(iolog)
    end

    @printf(iolog, "%s %s training complete\n", @timestr, procname)

    # fetch model from GPU device
    if !oncpu()
        setcpu()
        todevice!(m.net)
    end
end


function decode(m::Decoder, args::Dict, iolog=stdout)
    procname = @sprintf("NER[%s]", get!(args, "jobid", "-"))

    # read samples
    samples = load_samples(args["test_file"], m.words, m.chars, m.tags)

    # get args
    use_gpu = getarg!(args, "use_gpu", 0)
    batchsize = getarg!(args, "batchsize", 1)
    n_pred = length(samples)

    @printf(iolog, "%s %s decode - batchsize:%d n_pred:%d use_gpu:%s\n",
            @timestr, procname, batchsize, n_pred, string(use_gpu))

    # GPU device setup
    if use_gpu
        setcuda(0)
        todevice!(m.net)
    end

    # iterator
    pred_iter = SampleIterater(samples, batchsize, n_pred, shuffle=false, sort=false)

    settrain(false)
    preds = Array{Int, 2}(undef, 1, 0)
    probs = Array{Float32, 2}(undef, length(m.tags), 0)
    for (i, s) in enumerate(pred_iter)
        pred, prob = m.net(Float32, m.charvecs, m.wordvecs, s)
        preds = hcat(preds, reshape(pred, 1, :))
        probs = hcat(probs, prob)
    end

    @printf(iolog, "%s %s decode complete\n", @timestr, procname)

end

function getarg!(args::Dict, key::String, default::Any)
    (get!(args, key, default) == nothing) ? args[key] = default : args[key]
end
