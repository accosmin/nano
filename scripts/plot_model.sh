#!/bin/bash

# input file: *.state (train loss, train error, valid loss, valid error)
ifile=$1

# temporary gnuplot script file
pfile=${ifile/.state/.gnuplot}

# title
bifile=`basename ${ifile} .state`
title=${bifile//_/-}

# data attributes
tlindex=0
teindex=1
vlindex=2
veindex=3

labels=(`echo "train-loss train-error valid-loss valid-error"`)
styles=(`echo "lt:4:pt:1:lw:1:ps:0.3 lt:1:pt:2:lw:2:ps:0.3 lt:3:pt:3:lw:1:ps:0.3 lt:5:pt:6:lw:2:ps:0.3"`)

for format in `echo "svg pdf"`
do
        # output file
        ofile=${ifile/.state/.${format}}

        # set the plotting attributes
        rm -f ${pfile}
        echo "set terminal ${format} enhanced font ',7'" >> ${pfile}        
        echo "set output \"${ofile}\"" >> ${pfile}
        echo "set multiplot" >> ${pfile}
        echo "set size 1.0,1.0" >> ${pfile}

        # create plot: train vs validation (loss value and error)
        echo "set origin 0.0,0.5" >> ${pfile}
        echo "set size 1.0,0.5" >> ${pfile}
        echo "set title \"${title}\"" >> ${pfile}
        echo "set xlabel \"epochs/iterations\"" >> ${pfile}
        echo "set ylabel \"loss/error\"" >> ${pfile}
        echo "set xrange [*:*]" >> ${pfile}
        echo "set yrange [*:*]" >> ${pfile}
        echo "set xtic auto" >> ${pfile}
        echo "set ytic auto" >> ${pfile}   
        echo "set grid xtics ytics" >> ${pfile}
        echo "set key right top" >> ${pfile}

        echo -n "plot " >> ${pfile}
        echo -e "\t'${ifile}' using $((tlindex+1)) title '${labels[$tlindex]}' with linespoints ${styles[$tlindex]//:/ },\\" >> ${pfile}
        echo -e "\t'${ifile}' using $((teindex+1)) title '${labels[$teindex]}' with linespoints ${styles[$teindex]//:/ },\\" >> ${pfile}
        echo -e "\t'${ifile}' using $((vlindex+1)) title '${labels[$vlindex]}' with linespoints ${styles[$vlindex]//:/ },\\" >> ${pfile}
        echo -e "\t'${ifile}' using $((veindex+1)) title '${labels[$veindex]}' with linespoints ${styles[$veindex]//:/ }" >> ${pfile}
        echo "" >> ${pfile}

        # create plot: train vs validation error
        echo "set origin 0.0,0.0" >> ${pfile}
        echo "set size 0.5,0.5" >> ${pfile}
        echo "set title \"${title}\"" >> ${pfile}
        echo "set xlabel \"epochs/iterations\"" >> ${pfile}
        echo "set ylabel \"error\"" >> ${pfile}
        echo "set xrange [*:*]" >> ${pfile}
        echo "set yrange [*:*]" >> ${pfile}
        echo "set xtic auto" >> ${pfile}
        echo "set ytic auto" >> ${pfile}
        echo "set grid xtics ytics" >> ${pfile}
        echo "set key right top" >> ${pfile}

        echo -n "plot " >> ${pfile}
        echo -e "\t'${ifile}' using $((teindex+1)) title '${labels[$teindex]}' with linespoints ${styles[$teindex]//:/ },\\" >> ${pfile}
        echo -e "\t'${ifile}' using $((veindex+1)) title '${labels[$veindex]}' with linespoints ${styles[$veindex]//:/ }" >> ${pfile}
        echo "" >> ${pfile}

        # create plot: train vs validation loss value
        echo "set origin 0.5,0.0" >> ${pfile}
        echo "set size 0.5,0.5" >> ${pfile}
        echo "set title \"${title}\"" >> ${pfile}
        echo "set xlabel \"epochs/iterations\"" >> ${pfile}
        echo "set ylabel \"loss\"" >> ${pfile}
        echo "set xrange [*:*]" >> ${pfile}
        echo "set yrange [*:*]" >> ${pfile}
        echo "set xtic auto" >> ${pfile}
        echo "set ytic auto" >> ${pfile}   
        echo "set grid xtics ytics" >> ${pfile}
        echo "set key right top" >> ${pfile}

        echo -n "plot " >> ${pfile}
        echo -e "\t'${ifile}' using $((tlindex+1)) title '${labels[$tlindex]}' with linespoints ${styles[$tlindex]//:/ },\\" >> ${pfile}
        echo -e "\t'${ifile}' using $((vlindex+1)) title '${labels[$vlindex]}' with linespoints ${styles[$vlindex]//:/ }" >> ${pfile}
        echo "" >> ${pfile}

        # export
        echo "unset multiplot" >> ${pfile}
        gnuplot ${pfile} 
        rm -f ${pfile}
done
