#!/bin/bash

source $(dirname $0)/common_plot.sh

# check arguments
if [[ ($# -ne 1) || ("$1" == "help") || ("$1" == "--help") || ("$1" == "-h") ]]
then
        printf "plot_model.sh <training history file (.state)>\n"
        exit 1
fi

# input file with the following format:
#  ({train, valid, test} x {criterion, loss{average, variance, maximum}, error{average, variance, maximum}, time)+
ifile=$1

# temporary gnuplot script file
pfile=${ifile/.state/.gnuplot}

# title
bifile=$(basename ${ifile} .state)
title=${bifile//_/-}

# output file
ofile=${ifile/.state/.pdf}

# set the plotting attributes
rm -f ${pfile}
prepare_terminal ${pfile} ${ofile}

# plot each field ...
for ((i=0;i<7;i++))
do
        prepare_plot ${pfile} ${title} "epochs/iterations" ${ylabels[$i]}

        # compare results for the training, validation and test datasets
        printf "plot " >> ${pfile}
        printf "'%s' using %d title '%s' %s, " ${ifile} $(($i+1))  ${titles[$(($i+0))]}  "${train_style}" >> ${pfile}
        printf "'%s' using %d title '%s' %s, " ${ifile} $(($i+8))  ${titles[$(($i+7))]}  "${valid_style}" >> ${pfile}
        printf "'%s' using %d title '%s' %s\n" ${ifile} $(($i+15)) ${titles[$(($i+14))]} "${test_style}"  >> ${pfile}
done

# export
gnuplot ${pfile}
rm -f ${pfile}
