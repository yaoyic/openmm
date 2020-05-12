## OpenMM, patched with fast potential energy and force calculation routines.

Usage Example
------
```Python3
# Modified from OpenMM tutorial
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from timeit import default_timer as timer
import mdtraj as md
import numpy as np

PDB_FILENAME = 'alanine-dipeptide-nowater.pdb'
FRAMES_FILENAME = 'test_run_output.dcd'

# def system <- subject to change according to actual situation
pdb = PDBFile(PDB_FILENAME)
forcefield = ForceField('amber99sbildn.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff, constraints=HBonds)
integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
platform = Platform.getPlatformByName('CUDA') # <- all platforms are supported, but works the best on CUDA platform
simulation = Simulation(pdb.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)

# load coordinates <- e.g., from MD trajectory
frames = md.load(FRAMES_FILENAME, top=PDB_FILENAME)

# actual calculation
start = timer()
Es, Fs = simulation.context.EFcalc(frames.xyz)
stop = timer()
print("Batch method needs %.3fs time for 10000 points" % (stop - start))

# (optional, only for benchmarks)
Es2 = []
Fs2 = []
start = timer()
for posi in frames.xyz:
    simulation.context.setPositions(posi)
    state = simulation.context.getState(getEnergy=True, getForces=True)
    Es2.append(state.getPotentialEnergy()._value)
    Fs2.append(state.getForces(asNumpy=True)._value)
Es2 = np.array(Es2)
Fs2 = np.array(Fs2)
stop = timer()
print("Original method needs %.3fs time for 10000 points" % (stop - start))
# depending on the platform and system, bigger thershold might be necessary for np.allclose
if np.allclose(Es, Es2) and np.allclose(Fs, Fs2):
    print("Two methods give identical results!")
else:
    print("ERROR: Two methods give different results!")
```

---
## Original README
[![Build Status](https://travis-ci.org/openmm/openmm.svg?branch=master)](https://travis-ci.org/openmm/openmm?branch=master)
[![Anaconda Cloud Badge](https://anaconda.org/omnia/openmm/badges/downloads.svg)](https://anaconda.org/omnia/openmm)

## OpenMM: A High Performance Molecular Dynamics Library

Introduction
------------

[OpenMM](http://openmm.org) is a toolkit for molecular simulation. It can be used either as a stand-alone application for running simulations, or as a library you call from your own code. It
provides a combination of extreme flexibility (through custom forces and integrators), openness, and high performance (especially on recent GPUs) that make it truly unique among simulation codes.  

Getting Help
------------

Need Help? Check out the [documentation](http://docs.openmm.org/) and [discussion forums](https://simtk.org/forums/viewforum.php?f=161).
