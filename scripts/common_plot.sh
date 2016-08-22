#!/bin/bash

# data attributes
indices=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21)
index_time=22

protos="train valid test"

titles=""
ylabels=""

for proto in ${protos}
do
        titles=${titles}" ${proto}-criterion"
        titles=${titles}" ${proto}-loss-avg  ${proto}-loss-var  ${proto}-loss-max"
        titles=${titles}" ${proto}-error-avg ${proto}-error-var ${proto}-error-max"

        ylabels=${ylabels}" criterion"
        ylabels=${ylabels}" loss-avg  loss-var  loss-max"
        ylabels=${ylabels}" error-avg error-var error-max"
done

titles=(${titles})
ylabels=(${ylabels})

train_style=" with line lt 1 lw 1 lc rgb 'red'"
valid_style=" with line lt 3 lw 1 lc rgb 'blue'"
test_style=" with line lt 3 lw 1 lc rgb 'green'"

# create plot: train vs validation (loss value and error)
train_color=

function prepare_terminal
{
        __pfile=$1
        __ofile=$2

        # Should change here font name & size depending on the platform
        # NB: It is hard to find a font to generate good looking plots on both OSX & ArchLinux
        printf "set terminal pdfcairo enhanced color dashed font ',6'\n" >> ${__pfile}
        printf "set termoption dash\n" >> ${__pfile}
        printf "set output '%s'\n" ${__ofile} >> ${__pfile}
}

function prepare_plot
{
        __pfile=$1
        __title=$2
        __xlabel=$3
        __ylabel=$4

        printf "set autoscale\n" >> ${__pfile}
        printf "unset log\n" >> ${__pfile}
        printf "unset label\n" >> ${__pfile}
        printf "set xrange [*:*]\n" >> ${__pfile}
        printf "set yrange [*:*]\n" >> ${__pfile}
        printf "set xtic auto\n" >> ${__pfile}
        printf "set ytic auto\n" >> ${__pfile}
        printf "set grid xtics ytics\n" >> ${__pfile}
        printf "set key right top\n" >> ${__pfile}
        printf "set title '${__title}'\n" >> ${__pfile}
        printf "set xlabel '${__xlabel}'\n" >> ${__pfile}
        printf "set ylabel '${__ylabel}'\n" >> ${__pfile}
}

