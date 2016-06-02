#!/bin/bash

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

