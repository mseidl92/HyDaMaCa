# HyDaMaCa
The repository contains the code of *The Hydrodynamic Damping Matrix Calculator for Small Marine Craft Based on OpenFOAM* as well as the data used to create the results in the accompanying publication. If using HyDaMaCa for scientific research, please cite the related paper:
```
to be published
```

## software

The HyDaMaCa software can be used to predict drag coefficients of USVs, AUVs, and other small floating or submerged objects. The software is based on the opensource CFD framework OpenFOAM. A STL file of the object to be analyzed is required as input. The software offers a high-level UI that executes the analysis automatically. A low-level UI allows for running CFD experiments at desired velocities and obtaining raw data. For details on software prerequisites, required properties of the STL file and other design choices see the documentation. For a general software description and results of validation efforts see the accompanying publication.

## data

The 3D models turned into STL files for analysis to validate HyDaMaCa are all publically available:

- [NPS ARIES](https://savage.nps.edu/Savage/Robots/UnmannedUnderwaterVehicles/AriesNoThrusterPortsIndex.html)
- [WHOI REMUS 100](https://gitlab.nps.edu/me3720/simulation/remus_gazebo/-/blob/master/remus_description/mesh/remus_new.dae)
- [Maritime Robotics Otter](https://github.com/jhlenes/usv_simulator/blob/master/otter_description/meshes/otter/otter.dae)
