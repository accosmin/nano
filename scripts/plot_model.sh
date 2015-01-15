#!/bin/bash

source common_plot.sh

# input file: *.state (train loss, train error, valid loss, valid error)
ifile=$1

# temporary gnuplot script file
pfile=${ifile/.state/.gnuplot}

# title
bifile=`basename ${ifile} .state`
title=${bifile//_/-}

# data attributes
tlindex=0
teindex_avg=1
teindex_var=2
vlindex=3
veindex_avg=4
veindex_var=5

labels=(`echo "train-loss train-error train-error-var valid-loss valid-error valid-error-var"`)
styles=(`echo "lt:1:pt:1:lw:1:ps:0.3 lt:1:pt:2:lw:2:ps:0.3 lt:1:pt:4:lw:2:ps:0.3 lt:3:pt:1:lw:1:ps:0.3 lt:3:pt:2:lw:2:ps:0.3 lt:3:pt:4:lw:2:ps:0.3"`)

# output file
ofile=${ifile/.state/.pdf}

# set the plotting attributes
rm -f ${pfile}
prepare_terminal ${pfile}
echo "set output \"${ofile}\"" >> ${pfile}
echo "set multiplot" >> ${pfile}
echo "set size 1.0,1.0" >> ${pfile}

# create plot: train vs validation (loss value and error)
prepare_plot ${pfile}
echo "set origin 0.0,0.5" >> ${pfile}
echo "set size 1.0,0.5" >> ${pfile}
echo "set title \"${title}\"" >> ${pfile}
echo "set xlabel \"epochs/iterations\"" >> ${pfile}
echo "set ylabel \"loss/error\"" >> ${pfile}

echo -n "plot " >> ${pfile}
echo -e "\t'${ifile}' using $((tlindex+1)) title '${labels[$tlindex]}' with linespoints ${styles[$tlindex]//:/ },\\" >> ${pfile}
echo -e "\t'${ifile}' using $((teindex_avg+1)) title '${labels[$teindex_avg]}' with linespoints ${styles[$teindex_avg]//:/ },\\" >> ${pfile}
echo -e "\t'${ifile}' using $((teindex_var+1)) title '${labels[$teindex_var]}' with linespoints ${styles[$teindex_var]//:/ },\\" >> ${pfile}
echo -e "\t'${ifile}' using $((vlindex+1)) title '${labels[$vlindex]}' with linespoints ${styles[$vlindex]//:/ },\\" >> ${pfile}
echo -e "\t'${ifile}' using $((veindex_avg+1)) title '${labels[$veindex_avg]}' with linespoints ${styles[$veindex_avg]//:/ },\\" >> ${pfile}
echo -e "\t'${ifile}' using $((veindex_var+1)) title '${labels[$veindex_var]}' with linespoints ${styles[$veindex_var]//:/ }" >> ${pfile}
echo "" >> ${pfile}

# create plot: train vs validation loss value
prepare_plot ${pfile}
echo "set origin 0.0,0.0" >> ${pfile}
echo "set size 0.33,0.5" >> ${pfile}
echo "set title \"${title}\"" >> ${pfile}
echo "set xlabel \"epochs/iterations\"" >> ${pfile}
echo "set ylabel \"loss\"" >> ${pfile}

echo -n "plot " >> ${pfile}
echo -e "\t'${ifile}' using $((tlindex+1)) title '${labels[$tlindex]}' with linespoints ${styles[$tlindex]//:/ },\\" >> ${pfile}
echo -e "\t'${ifile}' using $((vlindex+1)) title '${labels[$vlindex]}' with linespoints ${styles[$vlindex]//:/ }" >> ${pfile}
echo "" >> ${pfile}

# create plot: train vs validation average error
prepare_plot ${pfile}
echo "set origin 0.33,0.0" >> ${pfile}
echo "set size 0.33,0.5" >> ${pfile}
echo "set title \"${title}\"" >> ${pfile}
echo "set xlabel \"epochs/iterations\"" >> ${pfile}
echo "set ylabel \"error\"" >> ${pfile}

echo -n "plot " >> ${pfile}
echo -e "\t'${ifile}' using $((teindex_avg+1)) title '${labels[$teindex_avg]}' with linespoints ${styles[$teindex_avg]//:/ },\\" >> ${pfile}
echo -e "\t'${ifile}' using $((veindex_avg+1)) title '${labels[$veindex_avg]}' with linespoints ${styles[$veindex_avg]//:/ }" >> ${pfile}
echo "" >> ${pfile}

# create plot: train vs validation error variance
prepare_plot ${pfile}
echo "set origin 0.66,0.0" >> ${pfile}
echo "set size 0.33,0.5" >> ${pfile}
echo "set title \"${title}\"" >> ${pfile}
echo "set xlabel \"epochs/iterations\"" >> ${pfile}
echo "set ylabel \"error\"" >> ${pfile}

echo -n "plot " >> ${pfile}
echo -e "\t'${ifile}' using $((teindex_var+1)) title '${labels[$teindex_var]}' with linespoints ${styles[$teindex_var]//:/ },\\" >> ${pfile}
echo -e "\t'${ifile}' using $((veindex_var+1)) title '${labels[$veindex_var]}' with linespoints ${styles[$veindex_var]//:/ }" >> ${pfile}
echo "" >> ${pfile}

# export
echo "unset multiplot" >> ${pfile}
gnuplot ${pfile} 
rm -f ${pfile}
