using Pkg


Pkg.develop(PackageSpec(name="Merlin", url="https://github.com/hshindo/Merlin.jl.git"))
Pkg.develop(PackageSpec(name="LightNLP", url="https://github.com/penguins-yatsuo/LightNLP.jl.git"))
Pkg.add("ArgParse")
Pkg.add("JLD2")
Pkg.add("FileIO")

julia merlin_lightnlp.jl --training --model=data/polymer-name.conv.model.jld2 --wordvec-file=C:\Users\YATSUO\AppData\Local\Temp\wordvec.hdf5 --train-file=C:\Users\YATSUO\AppData\Local\Temp\9248.train.bioes --test-file=C:\Users\YATSUO\AppData\Local\Temp\9248.test.bioes --jobid=9248 --logfile=log/nlp.2018-09.log --neural-network=CONV --nepochs=3 --batchsize=2 --learning-rate=0.005 --droprate=0.2 --winsize-c=2 --winsize-w=5
