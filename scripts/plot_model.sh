#!/bin/bash

source $(dirname $0)/common_plot.sh

# check arguments
if [[ ($# -ne 1) || ("$1" == "help") || ("$1" == "--help") || ("$1" == "-h") ]]
then
        printf "plot_model.sh <training history file (.state)>\n"
        exit 1
fi

# input file: ({train, valid, test} x {loss, error, error variance}, time)+
ifile=$1

# temporary gnuplot script file
pfile=${ifile/.state/.gnuplot}

# title
bifile=$(basename ${ifile} .state)
title=${bifile//_/-}

# data attributes
tloss_config="using 1 title 'train-loss' with line lt 1 lw 1 lc rgb 'red'"
vloss_config="using 4 title 'valid-loss' with line lt 3 lw 1 lc rgb 'blue'"
eloss_config="using 7 title 'test-loss' with line lt 3 lw 1 lc rgb 'green'"

terror_config="using 2 title 'train-error' with line lt 1 dt '-' lw 1 lc rgb 'red'"
verror_config="using 5 title 'valid-error' with line lt 3 dt '-' lw 1 lc rgb 'blue'"
eerror_config="using 8 title 'test-error' with line lt 3 dt '-' lw 1 lc rgb 'green'"

terror_var_config="using 3 title 'train-error-var' with line lt 1 dt '.' lw 1 lc rgb 'red'"
verror_var_config="using 6 title 'valid-error-var' with line lt 3 dt '.' lw 1 lc rgb 'blue'"
eerror_var_config="using 9 title 'test-error-var' with line lt 3 dt '.' lw 1 lc rgb 'green'"

# output file
ofile=${ifile/.state/.pdf}

# set the plotting attributes
rm -f ${pfile}
prepare_terminal ${pfile} ${ofile}

# create plot: train vs validation (loss value and error)
prepare_plot ${pfile} ${title} "epochs/iterations" "loss/error"

printf "plot " >> ${pfile}
printf "'%s' %s, " ${ifile} "${tloss_config}" >> ${pfile}
printf "'%s' %s, " ${ifile} "${vloss_config}" >> ${pfile}
printf "'%s' %s, " ${ifile} "${eloss_config}" >> ${pfile}
printf "'%s' %s, " ${ifile} "${terror_config}" >> ${pfile}
printf "'%s' %s, " ${ifile} "${verror_config}" >> ${pfile}
printf "'%s' %s\n" ${ifile} "${eerror_config}" >> ${pfile}

# create plot: train vs validation loss value
prepare_plot ${pfile} ${title} "epochs/iterations" "loss"

printf "plot " >> ${pfile}
printf "'%s' %s, " ${ifile} "${tloss_config}" >> ${pfile}
printf "'%s' %s, " ${ifile} "${vloss_config}" >> ${pfile}
printf "'%s' %s\n" ${ifile} "${eloss_config}" >> ${pfile}

# create plot: train vs validation average error
prepare_plot ${pfile} ${title} "epochs/iterations" "error"

printf "plot " >> ${pfile}
printf "'%s' %s, " ${ifile} "${terror_config}" >> ${pfile}
printf "'%s' %s, " ${ifile} "${verror_config}" >> ${pfile}
printf "'%s' %s\n" ${ifile} "${eerror_config}" >> ${pfile}

# create plot: train vs validation error variance
prepare_plot ${pfile} ${title} "epochs/iterations" "error variance"

printf "plot " >> ${pfile}
printf "'%s' %s, " ${ifile} "${terror_var_config}" >> ${pfile}
printf "'%s' %s, " ${ifile} "${verror_var_config}" >> ${pfile}
printf "'%s' %s\n" ${ifile} "${eerror_var_config}" >> ${pfile}

# export
gnuplot ${pfile}
rm -f ${pfile}
