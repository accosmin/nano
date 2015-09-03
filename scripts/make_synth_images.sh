#!/bin/bash

#font="DejaVu-Sans-Mono-Bold"
# -font ${font}

params=""
params=${params}" -channel RGBA"
params=${params}" -background transparent"
params=${params}" -undercolor transparent"
params=${params}" -fill black"
params=${params}" -pointsize 32"
params=${params}" -depth 32"
#params=${params}" -gravity center"

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


