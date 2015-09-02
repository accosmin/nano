#!/bin/bash

text="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
name="synth_alphabet"

#font="DejaVu-Sans-Mono-Bold"
# -font ${font}
# -gravity center

convert -fill black -pointsize 32 -background none label:${text} -flatten ${name}.png
