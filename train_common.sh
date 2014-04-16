#!/bin/bash

# common parameters
batch="opt=lbfgs,iters=1024,eps=1e-6"
minibatch="batch=1024,iters=256,eps=1e-6"
stoch="opt=sgd,epoch=64"

# train a model (model-type-name, model-parameters, trainer-type-name, trainer-parameters, configuration-name)
function fn_train
{
        if [[ $# -eq 4 ]]
        then 
                mfile=${dir_exp}/$1-$2-$4.model
                lfile=${dir_exp}/$1-$2-$4.log

                mconfig="--model $1"
                tconfig="--trainer $2 --trainer-params $3"
                pconfig=${param} 
        else
                mfile=${dir_exp}/$1-$3-$5.model
                lfile=${dir_exp}/$1-$3-$5.log

                mconfig="--model $1 --model-params $2"
                tconfig="--trainer $3 --trainer-params $4"
                pconfig=${param}
        fi      
        
        echo "running <${mconfig}> ..."
        echo "running <${tconfig}> ..."
        echo "running <${pconfig}> ..."
        time ${trainer} ${pconfig} ${mconfig} ${tconfig} --output ${mfile} > ${lfile}
        echo -e "\tlog saved to <${lfile}>"
        echo
}  

