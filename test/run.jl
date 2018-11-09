using LightNLP2, Merlin, ArgParse

function get_args()
    s = ArgParseSettings(autofix_names=true)

    @add_arg_table s begin
        "--neural-network"
            help = "neural network type (conv|lstm)"
            required = true
        "--model"
            help = "path of traind model file"
            required = true
        "--training"
            help = "do training"
            action = :store_true
        "--jobid"
            help = "job identification"
            default = "-"
        "--log"
            help = "log"
            default = "stderr"
        "--wordvec-file"
            help = "path of word embeds file (HDF5)"
            default = "wordvec.hdf5"
        "--train-file"
            help = "path of train data file (BIOES)"
            default = "train.bioes"
        "--test-file"
            help = "path of test data file (BIOES)"
            default = "test.bioes"
        "--epochs"
            help = "number of epochs"
            arg_type = Int
            default = 1
        "--batchsize"
            help = "batchsize"
            arg_type = Int
            default = 10
        "--fine-tuning"
            help = "fine tuning mode"
            action = :store_true
        "--learning-rate"
            help = "learning rate"
            arg_type = Float64
            default = 0.0005
        "--hidden_dims"
            help = "output dimensions for hidden layers"
            default = "128:128"
        "--droprate"
            help = "rate of dropout"
            arg_type = Float64
            default = 0.2
        "--winsize-c"
            help = "window (half kernel) size of character convolution"
            arg_type = Int
            default = 2
        "--winsize-w"
            help = "[Convolution only] window (half kernel) size of word convolution"
            arg_type = Int
            default = 5
        "--use-gpu"
            help = "Use GPU device"
            action = :store_true
        "--ntrain"
            help = "Count of training data"
            arg_type = Int
        "--ntest"
            help = "Count of test data"
            arg_type = Int
        "--tags"
            help = "List of Tags. (Colon seperated string)"
            default = "B:I:O:E:S"
        "--verbose", "-v"
            help = "print verbose output"
            action = :store_true
    end
    return parse_args(ARGS, s)
end

function main()
    args = get_args()

    modelfile = args["model"]
    iolog = (haskey(args, "logfile") && args["logfile"] != "stderr"
                ? open(args["logfile"], "a") : stderr)

    if args["training"]
        model = args["fine_tuning"] ? LightNLP2.Decoder(modelfile) : LightNLP2.Decoder()
        model.tags = split(args["tags"], ":")
        model.words, model.wordvecs, model.chars, model.charvecs = LightNLP2.load_embeds(args["wordvec_file"]; csize=20)

        train!(model, args, iolog)
        save(model, modelfile)
    else
        model = LightNLP2.Decoder(modelfile)
        results = decode(model, args, iolog)
        foreach(t -> println(stdout, t), results)
    end

    close(iolog)
end

main()
