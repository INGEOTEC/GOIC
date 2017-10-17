#!/bin/bash

nprocs=${nprocs:=32}
BSIZE=${BSIZE:=0}
score=${score:=macrorecall}


function run_gold() {
    train="$1"
    test="$2"
    s=$3
    k=3

    output=${output:-output}
    outname=${output}/$k-$s.$(basename $train .json.gz)
    mkdir -p $output
	
    if [ ! -f "$outname".params ]
    then
	[ ! -f "$outname".params.v1 ] && goic-params -o "$outname".params.v1 "$train" -k $k -s $s -n $nprocs -S $score
	goic-params -o "$outname".params "$train" -r "$outname".params.v1 -H -k $k -n $nprocs -S $score
    fi
    
    _outname=$outname
    for k in $(seq 0 $BSIZE)
    do
	if [ $k = 0 ]
	then
	    outname=$_outname
	else
	    outname=$_outname.k=$k
	fi

	if [ ! -f "$outname".model ]
	then
	    #classifier='{"type": "svm", "kernel": "rbf", "C": 1}'
	    classifier='{"type": "mlp", "hidden_layer_sizes": [32, 16]}'
	    #classifier='{"type": "knn", "n_neighbors": 11, "weights": "distance"}'
	    #classifier='{"type": "sgd", "loss": "hinge", "penalty": "l2"}'
	    #classifier='{"type": "sgd", "loss": "log", "penalty": "l2"}'
	    export classifier
	    goic-train $train -m $_outname.params -i $k -o $outname.model
	fi
	
	if [ -f $outname.model ]
	then
	    goic-predict $test -m $outname.model -o $outname.predicted.json
	fi

	goic-perf $test $outname.predicted.json -o $outname.results | python -mjson.tool
	echo "--- $outname.results ----"
	cat $outname.results

	if [ x$vspace = xyes ] && [ -f $outname.model ]
	then
	     goic-model $train -m $outname.model -o $outname.train.vspace.json
	     goic-model $test -m $outname.model -o $outname.test.vspace.json
	fi
    done
}

klass=${klass:=klass}
SLURM=${SLURM:=yes}
params=${params:='{}'}
export klass
export params
cmd="$1"
shift
case "$cmd" in
--run)
	;;

--run-gold)
	run_gold "$@"
	exit 0
	;;
*)
        echo "Usage 1: bash $0 --run"
        echo "Usage 3: bash $0 --run-gold TRAIN GOLD samples"
	echo "Environments:"
	echo "output=$output"
	echo "SLURM=$SLURM"
	exit 1
	;;
esac


train=${train:="train.json"}
test=${test:="test.json"}

samples=32
if [ $SLURM = yes ]
then
    srun -c$nprocs --mem-per-cpu=2000 bash -x $0 --run-gold "$train" "$test" "$samples"
else
    bash -x $0 --run-gold "$train" "$test" "$samples"
fi
