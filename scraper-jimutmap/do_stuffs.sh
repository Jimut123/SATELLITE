#!/bin/bash

du -s Kolkata

unzip -qq Kolkata_colab.zip
mv Kolkata_colab/* .
mv *.jpeg Kolkata
mv *.png Kolkata

du -s Kolkata
rm -rf Kolkata_colab.zip
rm -rf Kolkata

