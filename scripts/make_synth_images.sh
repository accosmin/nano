#!/bin/bash

#font="DejaVu-Sans-Mono-Bold"
# -font ${font}

params=""
params=${params}" -colorspace sRGB"
params=${params}" -background #00000000"
params=${params}" -undercolor #00000000"
params=${params}" -fill #010101FF"
params=${params}" -pointsize 32"
params=${params}" -density 196"
params=${params}" -gravity south"

flist="fonts.list"

# use all monospace fonts
convert -list font | grep -i mono | grep -i font: > ${flist}
while read font
do
    fname=${font/Font: /}
    echo ${fname}
    
    # digits
    text="0123456789"
    name="synth_digits_${fname}.png"

    rm -f ${name}
    convert ${params} -font ${fname} label:${text} ${name}

    # alphabet
    text="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    name="synth_alphabet_${fname}.png"

    rm -f ${name}
    convert ${params} -font ${fname} label:${text} ${name}    
done < ${flist}

rm -f ${flist}




