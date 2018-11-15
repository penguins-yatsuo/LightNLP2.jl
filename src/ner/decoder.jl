export Decoder, save, load, prepare_train!, train!, decode

import ProgressMeter, Formatting

using Formatting: printfmt, printfmtln
using JLD2: JLDWriteSession, jldopen, read, write
using Merlin: settrain, isnothing, isparam
using Merlin: shuffle!, gradient!, SGD
using Merlin.CUDA: getdevice, setdevice, synchronize

mutable struct Decoder
    words::Vector{String}
    wordvecs::Matrix{Float32}
    chars::Vector{Char}
    charvecs::Matrix{Float32}
    tags::Vector{String}
    net
end

function Decoder(fname::String="")
    if isfile(fname)
        load(fname)
    else
        Decoder(String[], Array{Float32, 2}(undef, 0, 0), Char[], Array{Float32, 2}(undef, 0, 0), String[], nothing)
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
        m.words, m.wordvecs = LightNLP2.read_wordvecs(args["wordvec_file"])
        m.chars, m.charvecs = LightNLP2.create_charvecs(m.words, csize=20)
    else
        # NOOP
    end

end


function train!(m::Decoder, args::Dict, iolog=stderr)
    procname = Formatting.format("NER[{1}]", get!(args, "jobid", "-"))

    # read samples
    train_samples = load_samples(args["train_file"], m.words, m.chars, m.tags)
    test_samples = load_samples(args["test_file"], m.words, m.chars, m.tags)

    # get args
    use_gpu = getarg!(args, "use_gpu", 0)
    epochs = getarg!(args, "epochs", 1)
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
    printfmtln(iolog, "{1} {2} model - {3}", @timestr, procname, string(m.net))
    printfmtln(iolog, "{1} {2} embeds - words:{3} chars:{4}", @timestr, procname,
            length(m.words), length(m.chars))
    printfmtln(iolog, "{1} {2} data - traindata:{3} testdata:{4} tags:{5}", @timestr, procname,
            length(train_samples), length(test_samples), string(m.tags))
    printfmtln(iolog, "{1} {2} train - epochs:{3} batchsize:{4} n_train:{5} n_test:{6} use_gpu:{7}",
            @timestr, procname, epochs, batchsize, n_train, n_test, string(use_gpu))
    flush(iolog)

    # GPU device setup
    if use_gpu
        setdevice(0)
        @ondevice(m.net)
    end

    # test data iterator
    test_iter = SampleIterater(test_samples, batchsize, n_test, shuffle=false, sort=true)

    # start train
    for epoch = 1:epochs
        printfmtln(stderr, "Epoch: {1}", epoch)
        printfmtln(iolog, "{1} {2} begin epoch {3}", @timestr, procname, epoch)

        # train
        settrain(true)
        train_iter = SampleIterater(train_samples, batchsize, n_train, shuffle=true, sort=true)

        progress = ProgressMeter.Progress(length(train_iter), desc="Train: ")
        opt.rate = args["learning_rate"] * batchsize / sqrt(batchsize) / (1 + 0.05*(epoch-1))

        loss::Float64 = 0
        for (i, s) in enumerate(train_iter)
            z = m.net(Float32, m.wordvecs, m.charvecs, s)
            params = gradient!(z)
            foreach(opt, filter(x -> isparam(x), params))
            loss += sum(@cpu(z.data)) / length(s)

            println("loss:", loss, " sum(z):", sum(@cpu(z.data)), "length(s):", length(s))

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
            z = @cpu m.net(Float32, m.wordvecs, m.charvecs, s)
            append!(preds, argmax(z))
            append!(golds, s.t)
            synchronize()
        end

        # evaluation of this epochs
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
    @oncpu(m.net)
end


function decode(m::Decoder, args::Dict, iolog=stderr)
    procname = Formatting.format("NER[{1}]", get!(args, "jobid", "-"))

    # read samples
    samples = load_samples(args["test_file"], m.words, m.chars, m.tags)

    # get args
    use_gpu = getarg!(args, "use_gpu", 0)
    n_pred = length(samples)

    printfmtln(iolog, "{1} {2} decode - n_pred:{3} use_gpu:{4}",
            @timestr, procname, n_pred, string(use_gpu))

    # GPU device setup
    if use_gpu
        setdevice(0)
        @ondevice(m.net)
    end

    # iterator
    pred_iter = SampleIterater(samples, 1, n_pred, shuffle=false, sort=false)
    progress = ProgressMeter.Progress(length(pred_iter), desc="Decode: ")

    settrain(false)
    preds = Array{Int, 2}(undef, 1, 0)
    probs = Array{Float32, 2}(undef, length(m.tags), 0)
    for (i, s) in enumerate(pred_iter)
        z = @cpu m.net(Float32, m.wordvecs, m.charvecs, s)
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
