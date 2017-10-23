#!/bin/bash

nprocs=${nprocs:=32}
BSIZE=${BSIZE:=0}
score=${score:=macrorecall}

#export classifier='{"type": "", "hidden_layer_sizes": [16, 8]}'
#export classifier='{"type": "mlp", "hidden_layer_sizes": [32, 16]}'
#export classifier='{"type": "knn", "n_neighbors": 3, "weights": "distance"}'
#export classifier='{"type": "sgd", "loss": "log", "penalty": "l2"}'

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
	    __outname=$_outname
	else
	    __outname=$_outname.k=$k
	fi

	for classifier in '{"type": "linearsvm"}' \
	    '{"type": "svm", "kernel": "rbf", "C": 1}' \
	    '{"type": "svm", "kernel": "rbf", "C": 10}' \
	    '{"type": "svm", "kernel": "rbf", "C": 0.1}' \
	    '{"type": "mlp", "hidden_layer_sizes": [32, 16]}' \
	    '{"type": "knn", "n_neighbors": 11, "weights": "distance"}' \
	    '{"type": "sgd", "loss": "hinge", "penalty": "l2"}' \
	    '{"type": "sgd", "loss": "log", "penalty": "l2"}'
	do
	    suffix=$(echo "$classifier" | perl -ne 's/["\{\}\s,:]+/_/g; print')
	    outname=$__outname.$suffix
	    
	    export classifier
	    if [ ! -f "$outname".model ]
	    then
		goic-train $train -m $_outname.params -i $k -o $outname.model
	    fi
	    
	    if [ -f $outname.model ]
	    then
		goic-predict $test -m $outname.model -o $outname.predicted.json
	    fi
	    
	    goic-perf $test $outname.predicted.json -o $outname.results
	    echo "### $outname.results ###"
	    cat $outname.results | python -mjson.tool

	    echo "### execute the following commands to convert both the training and test set ###"
	    echo goic-model $train -m $outname.model -o $outname.train.vspace.json
	    echo goic-model $test -m $outname.model -o $outname.test.vspace.json
	done
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
