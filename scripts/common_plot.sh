#!/bin/bash

function prepare_terminal
{
        __pfile=$1

        # Should change here font name & size depending on the platform
        # NB: It is hard to find a font to generate good looking plots on both OSX & ArchLinux
        echo "set terminal pdfcairo enhanced color font ',6'" >> ${__pfile}
}

function prepare_plot
{
        __pfile=$1

        echo "set autoscale" >> ${__pfile}
        echo "unset log" >> ${__pfile}
        echo "unset label" >> ${__pfile}
        echo "set xrange [*:*]" >> ${__pfile}
        echo "set yrange [*:*]" >> ${__pfile}
        echo "set xtic auto" >> ${__pfile}
        echo "set ytic auto" >> ${__pfile}
        echo "set grid xtics ytics" >> ${__pfile}
        echo "set key right top" >> ${__pfile}
}

