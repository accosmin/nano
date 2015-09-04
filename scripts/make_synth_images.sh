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

# digits
text="0123456789"
name="synth_digits.png"

rm -f ${name}
convert ${params} label:${text} ${name}

# alphabet
text="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
name="synth_alphabet.png"

rm -f ${name}
convert ${params} label:${text} ${name}


