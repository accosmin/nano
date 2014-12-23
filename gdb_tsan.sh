echo $@

gdb -ex 'set disable-randomization off' --args $@
