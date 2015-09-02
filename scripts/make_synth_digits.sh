#!/bin/bash

text="0123456789"
name="synth_digits"

#font="DejaVu-Sans-Mono-Bold"
# -font ${font}
# -gravity center

convert -fill black -pointsize 32 -background none label:${text} -flatten ${name}.png
