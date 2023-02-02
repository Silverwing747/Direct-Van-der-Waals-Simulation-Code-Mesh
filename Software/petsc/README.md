# README #

### Tianyi Modified Version - Adaptive Time Scheme ###
* PETSC modified to read/export initial time derivative for restart simulation
* PETSC modified to implement TianyiAdaptTime1: When simulation diverged, reduce current timestep by the factor of TianyiAdaptFactor1;
* PETSC modified to implement TianyiAdaptTime2: When total linear/nonlinear iterations outside of desired range, increase or decrease timestep by the factor of TianyiAdaptFactor2;
* Following code is modified
- src/ts/impls/implicit/alpha/alpha1.c -> Implemented TianyiAdaptTime2 & Read initial time derivative
- src/ts/adpat/interface/tsadapt.c -> Implemented TianyiAdaptTime1
- src/ts/interface/tscreate.c -> Declare initial value for the new variable in TS structure
- include/petsc/private/tsimpl.h -> Define new variable in TS structure

## Arch Version explaination
* arch_debug -- Only use for debug purpose
* arch_original -- The unmodified version of petsc
* arch_TianyiAdaptive -- The petsc with Tianyi Modified adaptive timestep --with-debugging=no
* arch_TianyiAdaptive_opt -- Optimized petsc with Tianyi Modified adaptive timestep --with-debugging=no

### What is this repository for? ###

Host the PETSc numerical library package. http://www.mcs.anl.gov/petsc

### How do I get set up? ###

* Download http://www.mcs.anl.gov/petsc/download/index.html
* Install http://www.mcs.anl.gov/petsc/documentation/installation.html

### Contribution guidelines ###

* See the file CONTRIBUTING
* https://gitlab.com/petsc/petsc/wikis/Home

### Who do I talk to? ###

* petsc-maint@mcs.anl.gov
* http://www.mcs.anl.gov/petsc/miscellaneous/mailing-lists.html
