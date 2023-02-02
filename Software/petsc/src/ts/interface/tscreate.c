#include <petsc/private/tsimpl.h>      /*I "petscts.h"  I*/

const char *const TSConvergedReasons_Shifted[] = {
  "ADJOINT_DIVERGED_LINEAR_SOLVE",
  "FORWARD_DIVERGED_LINEAR_SOLVE",
  "DIVERGED_STEP_REJECTED",
  "DIVERGED_NONLINEAR_SOLVE",
  "CONVERGED_ITERATING",
  "CONVERGED_TIME",
  "CONVERGED_ITS",
  "CONVERGED_USER",
  "CONVERGED_EVENT",
  "CONVERGED_PSEUDO_FATOL",
  "CONVERGED_PSEUDO_FATOL",
  "TSConvergedReason","TS_",0};
const char *const*TSConvergedReasons = TSConvergedReasons_Shifted + 4;

/*@C
  TSCreate - This function creates an empty timestepper. The problem type can then be set with TSSetProblemType() and the
       type of solver can then be set with TSSetType().

  Collective

  Input Parameter:
. comm - The communicator

  Output Parameter:
. ts   - The TS

  Level: beginner

  Developer Notes:
    TS essentially always creates a SNES object even though explicit methods do not use it. This is
                    unfortunate and should be fixed at some point. The flag snes->usessnes indicates if the
                    particular method does use SNES and regulates if the information about the SNES is printed
                    in TSView(). TSSetFromOptions() does call SNESSetFromOptions() which can lead to users being confused
                    by help messages about meaningless SNES options.

.seealso: TSSetType(), TSSetUp(), TSDestroy(), TSSetProblemType()
@*/
PetscErrorCode  TSCreate(MPI_Comm comm, TS *ts)
{
  TS             t;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(ts,1);
  *ts = NULL;
  ierr = TSInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(t, TS_CLASSID, "TS", "Time stepping", "TS", comm, TSDestroy, TSView);CHKERRQ(ierr);

  /* General TS description */
  t->problem_type      = TS_NONLINEAR;
  t->equation_type     = TS_EQ_UNSPECIFIED;

  t->ptime             = 0.0;
  t->time_step         = 0.1;
  t->max_time          = PETSC_MAX_REAL;
  t->exact_final_time  = TS_EXACTFINALTIME_UNSPECIFIED;
  t->steps             = 0;
  t->max_steps         = PETSC_MAX_INT;
  t->steprestart       = PETSC_TRUE;

  t->max_snes_failures = 1;
  t->max_reject        = 10;
  t->errorifstepfailed = PETSC_TRUE;

  t->rhsjacobian.time  = PETSC_MIN_REAL;
  t->rhsjacobian.scale = 1.0;
  t->ijacobian.shift   = 1.0;

  /* All methods that do adaptivity should specify
   * its preferred adapt type in their constructor */
  t->default_adapt_type = TSADAPTNONE;
  t->atol               = 1e-4;
  t->rtol               = 1e-4;
  t->cfltime            = PETSC_MAX_REAL;
  t->cfltime_local      = PETSC_MAX_REAL;

  t->num_rhs_splits     = 0;

  // Tianyi Added Parameter
  t->recover            = PETSC_FALSE;

  t->time_next_print	= 0.0; // Desired print out time.
  t->TianyiAdjustFactor1	= 0.5; // First time adjustment for adaptive time step, smaller than 1
  t->TianyiAdjustFactor2	= 0.1; // Second time adjustment for adaptive time step, larger than 0

  t->min_nits			= 2; // minimum wanted snes iteration number
  t->min_lits			= 200; // minimum wanted ksp iteration number
  t->max_nits			= 5; // maximum wanted snes iteration number
  t->max_lits			= 1000; // maximum wanted ksp iteration number

  t->TianyiAdaptTime1	= PETSC_FALSE; // Flag for first time adjustment or not;
  t->TianyiAdaptTime2	= PETSC_FALSE; // Flag for second time adjustment or not;

  *ts = t;
  PetscFunctionReturn(0);
}
