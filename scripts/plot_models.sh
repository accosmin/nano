#!/bin/bash

# input files: [output plot] [*.state (train loss, train error, valid loss, valid error)]+
ifiles=("$@")
  
if [ ${#ifiles[@]} -lt 2 ]
then
        echo "plot_models.sh <output path> <training log file>+"
        exit
fi

# output file
ofile=${ifiles[0]}

# temporary gnuplot script file
pfile=${ofile/.pdf/.gnuplot}

# data attributes
indices=(1 2 3 4)
titles=(`echo "train-loss train-error valid-loss valid-error"`)
origins=(`echo "0.0,0.0 0.5,0.0 0.0,0.5 0.5,0.5"`)
sizes=(`echo "0.5,0.5 0.5,0.5 0.5,0.5 0.5,0.5"`)

# set the plotting attributes
rm -f ${pfile}
echo "set terminal pdf enhanced font ',7'" >> ${pfile}        
echo "set output \"${ofile}\"" >> ${pfile}
echo "set multiplot" >> ${pfile}
echo "set size 1.0,1.0" >> ${pfile}

# create sub-plots for each data type
for ((k=0;k<${#indices[*]};k++))
do
        index=${indices[$k]}
        origin=${origins[$k]}
        size=${sizes[$k]}
        title=${titles[$k]}

        echo "set origin ${origin}" >> ${pfile}
        echo "set size ${size}" >> ${pfile}
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
        for ((i=1;i<${#ifiles[*]};i++))
        do
                ifile=${ifiles[$i]}
                label=`basename ${ifile} .state`
                label=${label//_/-}
                
                echo -e -n "\t'${ifile}' using ${index} title '${label}' with linespoints ps 0.3" >> ${pfile}
                
                let ii=${i}+1
                if [ $ii -eq ${#ifiles[*]} ] 
                then
                        echo "" >> ${pfile}
                else
                        echo ",\\" >> ${pfile}
                fi
        done
        echo "" >> ${pfile}
done

# export
echo "unset multiplot" >> ${pfile}
gnuplot ${pfile} 
rm -f ${pfile}
