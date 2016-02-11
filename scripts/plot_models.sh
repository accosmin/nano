#!/bin/bash

source $(dirname $0)/common_plot.sh

# check arguments
if [[ ($# -lt 2) || ("$1" == "help") || ("$1" == "--help") || ("$1" == "-h") ]]
then
        printf "plot_models.sh <output path (.pdf)> <training state files (.state)>\n"
        exit 1
fi

# input files: (train loss, train error, train error variance, valid loss, valid error, valid error variance, time)+
ifiles=("$@")
inames=("$@")
isize=${#ifiles[*]}

for ((i=1;i<${isize};i++))
do
        ifile=${ifiles[$i]}
        label=$(basename ${ifile} .state)
        inames[$i]=${label//_/-}
done

# output file
ofile=${ifiles[0]}

# format (extension)
format="${ofile##*.}"

# temporary gnuplot script file
pfile=${ofile/.${format}/.gnuplot}

# data attributes
indices=(1 2 3 4 5 6)
titles=(`echo  "train-loss train-error train-error-var valid-loss valid-error valid-error-var"`)

# set the plotting attributes
rm -f ${pfile}
prepare_terminal ${pfile} ${ofile}

# create plots for each data type
for ((k=0;k<${#indices[*]};k++))
do
        index=${indices[$k]}
        title=${titles[$k]}

        # plot against number of training epochs
        prepare_plot ${pfile} ${title} "epochs/iterations" "loss/error"
        printf "plot " >> ${pfile}
        for ((i=1;i<${isize};i++))
        do
                ifile=${ifiles[$i]}
                iname=${inames[$i]}

                printf "'%s' using %d title '%s' with line" "${ifiles[$i]}" ${index} "${inames[$i]}" >> ${pfile}
                [[ $(($i+1)) != ${isize} ]] && printf ", " >> ${pfile}
        done
        printf "\n" >> ${pfile}

        # plot against training time
        prepare_plot ${pfile} ${title} "seconds" "loss/error"
        printf "plot " >> ${pfile}
        for ((i=1;i<${isize};i++))
        do
                ifile=${ifiles[$i]}
                iname=${inames[$i]}

                printf "'%s' using 7:%d title '%s' with line" "${ifiles[$i]}" ${index} "${inames[$i]}" >> ${pfile}
                [[ $(($i+1)) != ${isize} ]] && printf ", " >> ${pfile}
        done
        printf "\n" >> ${pfile}
done

# export
gnuplot ${pfile}
rm -f ${pfile}
