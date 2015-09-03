#!/bin/bash

#font="DejaVu-Sans-Mono-Bold"
# -font ${font}
# -gravity center

params=""
params=${params}" -background rgba(0,0,0,0)"
#params=${params}" -undercolor rgba(0,0,0,0)"
params=${params}" -fill rgba(0,0,0,255)"
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


