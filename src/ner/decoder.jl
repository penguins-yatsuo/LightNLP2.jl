export Decoder, save, load, prepare_train!, train!, decode

import ProgressMeter

using Formatting: printfmt, printfmtln
using JLD2: JLDWriteSession, jldopen, read, write
using Merlin: settrain, isnothing, isparam, gradient!, SGD
using Merlin.CUDA: getdevice, synchronize

mutable struct Decoder
    words::Vector{String}
    chars::Vector{Char}
    tags::Vector{String}
    net
end

function Decoder(fname::String="")
    if isfile(fname)
        load(fname)
    else
        Decoder(String[], Char[], String[], nothing)
    end
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

function prepare_train!(m::Decoder, args::Dict)

    if args["init_model"]
        m.tags = split(args["tags"], ":")
        m.words = LightNLP2.embed_words(args["wordvec_file"])
        m.chars = LightNLP2.embed_chars(m.words)
        m.net = nothing

    else
        tags = split(args["tags"], ":")
        if m.tags != tags
            @info "Tags changed. Initialize output layer."
            m.tags = tags
            if typeof(m.net) in [ConvNet, LstmNet]
                init_output!(m.net, length(m.tags))
            end
        end

    end

end


function train!(m::Decoder, args::Dict, iolog=stderr)
    procname = Formatting.format("NER[{1}]", get!(args, "jobid", "-"))

    # vectors
    m.words, wordvecs = embed_wordvecs!(m.words, args["wordvec_file"])
    m.chars, charvecs = embed_charvecs!(m.chars, csize=20)

    # read samples
    train_samples = load_samples(args["train_file"], m.words, m.chars, m.tags)
    test_samples = load_samples(args["test_file"], m.words, m.chars, m.tags)

    # get args
    cpu_only = getarg!(args, "cpu", false)
    epochs = getarg!(args, "epochs", 1)
    batchsize = getarg!(args, "batchsize", 1)
    ntags = getarg!(args, "ntags", length(m.tags))
    n_train = getarg!(args, "ntrain", length(train_samples))
    n_test = getarg!(args, "ntest", length(test_samples))

    # create neural network
    if m.net == nothing
        m.net = begin
            lowercase(args["neural_network"]) == "conv" ? ConvNet(args, wordvecs, charvecs, ntags) :
            lowercase(args["neural_network"]) == "lstm" ? LstmNet(args, wordvecs, charvecs, ntags) : nothing
        end
    end

    # optimizer
    opt = SGD()

    # print job status
    printfmtln(iolog, "{1} {2} model - {3}", @timestr, procname, string(m.net))
    printfmtln(iolog, "{1} {2} embeds - words:{3} chars:{4}", @timestr, procname,
            length(m.words), length(m.chars))
    printfmtln(iolog, "{1} {2} data - traindata:{3} testdata:{4} tags:{5}", @timestr, procname,
            length(train_samples), length(test_samples), string(m.tags))
    printfmtln(iolog, "{1} {2} train - epochs:{3} batchsize:{4} n_train:{5} n_test:{6} cpu_only:{7}",
            @timestr, procname, epochs, batchsize, n_train, n_test, string(cpu_only))
    flush(iolog)

    # device setup
    @setdevice(cpu_only ? CPU : getdevice())
    @device m.net

    # test data iterator
    test_iter = SampleIterater(test_samples, batchsize, n_test, shuffle=false, sort=true)

    # start train
    decay = getarg!(args, "decay", 0.0001)
    opt.rate = getarg!(args, "learning_rate", 0.0005)
    for epoch = 1:epochs
        printfmtln(stderr, "Epoch: {1}", epoch)
        printfmtln(iolog, "{1} {2} begin epoch {3}", @timestr, procname, epoch)

        # time-based decay
        opt.rate = opt.rate * (1.0 / (1.0 + decay * (epoch - 1)))

        # train
        settrain(true)
        train_iter = SampleIterater(train_samples, batchsize, n_train, shuffle=true, sort=true)
        progress = ProgressMeter.Progress(length(train_iter), desc="Train: ")

        loss::Float64 = 0
        for (i, s) in enumerate(train_iter)
            z = m.net(Float32, s)
            params = gradient!(z)
            foreach(opt, filter(x -> isparam(x), params))
            loss += sum(@host(z).data) / length(s)
            ProgressMeter.next!(progress)
            synchronize()
        end
        loss /= length(train_iter)
        ProgressMeter.finish!(progress)
        printfmt(stderr, "Loss: {1:.5f} ", loss)

        # test
        settrain(false)
        preds = Int[]
        golds = Int[]
        for (i, s) in enumerate(test_iter)
            z = @host m.net(Float32, s)
            append!(preds, argmax(z))
            append!(golds, s.t)
            synchronize()
        end

        # evaluation of this epochs
        println(m.tags)
        span_golds = span_decode(golds, m.tags)
        span_preds = span_decode(preds, m.tags)
        
        prec, recall, fval = fscore(span_golds, span_preds)

        printfmtln(stderr, "Prec: {1:.5f} Recall: {2:.5f} Fscore: {3:.5f}", prec, recall, fval)
        printfmtln(iolog, "{1} {2} end epoch {3} - loss:{4:.5f} prec:{5:.5f} recall:{6:.5f} fscore:{7:.5f}",
            @timestr, procname, epoch, loss, prec, recall, fval)
        flush(iolog)
    end

    printfmtln(iolog, "{1} {2} training complete", @timestr, procname)
    flush(iolog)

    # fetch model from GPU device
    @host m.net
end


function decode(m::Decoder, args::Dict, iolog=stderr)
    procname = Formatting.format("NER[{1}]", get!(args, "jobid", "-"))

    # read samples
    samples = load_samples(args["test_file"], m.words, m.chars, m.tags)

    # get args
    cpu_only = getarg!(args, "cpu", false)
    n_pred = length(samples)

    printfmtln(iolog, "{1} {2} decode - n_pred:{3} cpu_only:{4}",
            @timestr, procname, n_pred, string(cpu_only))

    # device setup
    @setdevice(cpu_only ? CPU : getdevice())
    @device m.net

    # iterator
    pred_iter = SampleIterater(samples, 1, n_pred, shuffle=false, sort=false)
    progress = ProgressMeter.Progress(length(pred_iter), desc="Decode: ")

    settrain(false)
    preds = Array{Int, 2}(undef, 1, 0)
    probs = Array{Float32, 2}(undef, length(m.tags), 0)
    for (i, s) in enumerate(pred_iter)
        z = @host m.net(Float32, s)
        preds = hcat(preds, reshape(argmax(z), 1, :))
        probs = hcat(probs, z.data)
        ProgressMeter.next!(progress)
        synchronize()
    end
    ProgressMeter.finish!(progress)
    printfmtln(iolog, "`{1} {2} decode complete", @timestr, procname)

    merge_decode(readlines(args["test_file"]), m.tags, preds, probs)
end

function getarg!(args::Dict, key::String, default::Any)
    (get!(args, key, default) == nothing) ? args[key] = default : args[key]
end
