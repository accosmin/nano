#!/bin/bash

params=""
params=${params}" -colorspace sRGB"
params=${params}" -background #00000000"
params=${params}" -undercolor #00000000"
params=${params}" -fill #010101FF"
params=${params}" -pointsize 16"
params=${params}" -density 196"
params=${params}" -gravity south"

text=""
text=${text}"0123456789"
text=${text}"abcdefghijklmnopqrstuvwxyz"
text=${text}"ABCDEFGHIJKLMNOPQRSTUVWXYZ"

echo "characters ${text}"

flist="fonts.list"

# use all available monospace fonts
convert -list font | grep -i mono | grep -vi italic | grep -vi oblique | grep -i font: > ${flist}
while read fname
do
    font=${fname/Font: /}
    echo "using font ${font} ..."

    ipath="synth_${font}.png"

    rm -f ${ipath}
    convert ${params} -font ${font} label:${text} ${ipath}
done < ${flist}

rm -f ${flist}
