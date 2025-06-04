#!/bin/bash

cd ./temp
source /opt/openfoam10/etc/bashrc
blockMesh
surfaceFeatures
snappyHexMesh -overwrite
pimpleFoam
