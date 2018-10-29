using ArgParse

using LightNLP2
using JLD2, FileIO

function get_args()
    s = ArgParseSettings(
        autofix_names=true)

    @add_arg_table s begin
        "--neural-network"
            help = "neural network type (conv|lstm)"
            required = true
        "--model"
            help = "path of traind model file (JLD2)"
            required = true
        "--training"
            help = "do training"
            action = :store_true
        "--jobid"
            help = "job identification"
            default = "-"
        "--logfile"
            help = "logfile"
            default = "stdout"
        "--wordvec-file"
            help = "path of word embeds file (HDF5)"
            default = "wordvec.hdf5"
        "--train-file"
            help = "path of train data file (CONLL)"
            default = "train.bioes"
        "--test-file"
            help = "path of test data file (CONLL)"
            default = "test.bioes"
        "--nepochs"
            help = "number of epochs"
            arg_type = Int
            default = 1
        "--batchsize"
            help = "batchsize"
            arg_type = Int
            default = 10
        "--learning-rate"
            help = "learning rate"
            arg_type = Float64
            default = 0.0005
        "--nlayers"
            help = "number of neural netowork layers"
            arg_type = Int
            default = 1
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
        "--verbose", "-v"
            help = "print verbose output"
            action = :store_true
    end
    return parse_args(ARGS, s)
end

function main()

    args = get_args()

    modelfile = args["model"]

    iolog = (haskey(args, "logfile") && args["logfile"] != "stdout" 
                ? open(args["logfile"], "a") : stdout)

    if args["training"]

        decoder = LightNLP2.NER.Decoder(args, iolog)
        save(modelfile, "decoder", decoder)    

    else

        decoder = load(modelfile, "decoder")
        LightNLP.NER.decode(decoder, args)

    end

    close(iolog)
end

main()