#!/bin/bash

source common_plot.sh

# input files: [output plot] [*.state (train loss, train error, train error variance, valid loss, valid error, valid error variance)]+
ifiles=("$@")

if [ ${#ifiles[@]} -lt 2 ]
then
        echo "plot_models.sh <output path> <training log file>+"
        exit
fi

# output file
ofile=${ifiles[0]}

# format (extension)
format="${ofile##*.}"

# temporary gnuplot script file
pfile=${ofile/.${format}/.gnuplot}

# data attributes
indices=(1 2 3 4 5 6)
titles=(`echo "train-loss train-error train-error-var valid-loss valid-error valid-error-var"`)

# set the plotting attributes
rm -f ${pfile}
prepare_terminal ${pfile}
echo "set output \"${ofile}\"" >> ${pfile}
echo "set size 1.0,1.0" >> ${pfile}

# create plots for each data type
for ((k=0;k<${#indices[*]};k++))
do
        index=${indices[$k]}
        title=${titles[$k]}

        prepare_plot ${pfile}
        echo "set origin 0.0,0.0" >> ${pfile}
        echo "set size 1.0,1.0" >> ${pfile}
        echo "set title \"${title}\"" >> ${pfile}
        echo "set xlabel \"epochs/iterations\"" >> ${pfile}
        echo "set ylabel \"loss/error\"" >> ${pfile}

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
gnuplot ${pfile}
rm -f ${pfile}
