#include <petsc/private/petschpddm.h> /*I "petscksp.h" I*/
/* access to same_local_solves */
#include <../src/ksp/pc/impls/bjacobi/bjacobi.h>
#include <../src/ksp/pc/impls/asm/asm.h>

/* static array length */
#define ALEN(a) (sizeof(a)/sizeof((a)[0]))

static const char *HPDDMType[]              = { "gmres", "bgmres", "cg", "bcg", "gcrodr", "bgcrodr", "bfbcg" };
static const char *HPDDMOrthogonalization[] = { "cgs", "mgs" };
static const char *HPDDMQR[]                = { "cholqr", "cgs", "mgs" };
static const char *HPDDMVariant[]           = { "left", "right", "flexible" };
static const char *HPDDMRecycleTarget[]     = { "SM", "LM", "SR", "LR", "SI", "LI" };
static const char *HPDDMRecycleStrategy[]   = { "A", "B" };

static PetscBool citeKSP = PETSC_FALSE;
static const char hpddmCitationKSP[] = "@inproceedings{jolivet2016block,\n\tTitle = {{Block Iterative Methods and Recycling for Improved Scalability of Linear Solvers}},\n\tAuthor = {Jolivet, Pierre and Tournier, Pierre-Henri},\n\tOrganization = {IEEE},\n\tYear = {2016},\n\tSeries = {SC16},\n\tBooktitle = {Proceedings of the 2016 International Conference for High Performance Computing, Networking, Storage and Analysis}\n}\n";

static PetscErrorCode KSPSetFromOptions_HPDDM(PetscOptionItems *PetscOptionsObject, KSP ksp)
{
  KSP_HPDDM      *data = (KSP_HPDDM*)ksp->data;
  PetscInt       i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  data->scntl[0] = ksp->max_it;
  data->rcntl[0] = ksp->rtol;
  ierr = PetscOptionsHead(PetscOptionsObject, "KSPHPDDM options, cf. https://github.com/hpddm/hpddm");CHKERRQ(ierr);
  i = HPDDM_KRYLOV_METHOD_GMRES;
  ierr = PetscOptionsEList("-ksp_hpddm_type", "Type of Krylov method", "KSPHPDDM", HPDDMType, ALEN(HPDDMType), HPDDMType[HPDDM_KRYLOV_METHOD_GMRES], &i, NULL);CHKERRQ(ierr);
  data->cntl[5] = i;
  if (data->cntl[5] == HPDDM_KRYLOV_METHOD_RICHARDSON) {
    data->rcntl[0] = 1.0;
    ierr = PetscOptionsReal("-ksp_richardson_scale", "Damping factor used in Richardson iterations", "KSPHPDDM", data->rcntl[0], data->rcntl, NULL);CHKERRQ(ierr);
  } else {
    i = HPDDM_VARIANT_LEFT;
    if (ksp->pc_side_set == PC_SIDE_DEFAULT) {
      ierr = PetscOptionsEList("-ksp_hpddm_variant", "Left, right, or variable preconditioning", "KSPHPDDM", HPDDMVariant, ALEN(HPDDMVariant), HPDDMVariant[HPDDM_VARIANT_LEFT], &i, NULL);CHKERRQ(ierr);
    } else if (ksp->pc_side_set == PC_RIGHT) i = HPDDM_VARIANT_RIGHT;
    else if (ksp->pc_side_set == PC_SYMMETRIC) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Symmetric preconditioning not implemented");
    if (i != HPDDM_VARIANT_LEFT && (data->cntl[5] == HPDDM_KRYLOV_METHOD_BCG || data->cntl[5] == HPDDM_KRYLOV_METHOD_BFBCG)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Right and flexible preconditioned (BF)BCG not implemented");
    data->cntl[1] = i;
    if (i > 0) {
      ierr = KSPSetPCSide(ksp, PC_RIGHT);CHKERRQ(ierr);
    }
    if (data->cntl[5] == HPDDM_KRYLOV_METHOD_BGMRES || data->cntl[5] == HPDDM_KRYLOV_METHOD_BGCRODR || data->cntl[5] == HPDDM_KRYLOV_METHOD_BFBCG) {
      data->rcntl[1] = -1.0;
      ierr = PetscOptionsReal("-ksp_hpddm_deflation_tol", "Tolerance when deflating right-hand sides inside block methods", "KSPHPDDM", data->rcntl[1], data->rcntl + 1, NULL);CHKERRQ(ierr);
      i = 1;
      ierr = PetscOptionsRangeInt("-ksp_hpddm_enlarge_krylov_subspace", "Split the initial right-hand side into multiple vectors", "KSPHPDDM", i, &i, NULL, 1, std::numeric_limits<unsigned short>::max() - 1);CHKERRQ(ierr);
      data->scntl[1 + (data->cntl[5] != HPDDM_KRYLOV_METHOD_BFBCG)] = i;
    } else data->scntl[2] = 0;
    if (data->cntl[5] == HPDDM_KRYLOV_METHOD_GMRES || data->cntl[5] == HPDDM_KRYLOV_METHOD_BGMRES || data->cntl[5] == HPDDM_KRYLOV_METHOD_GCRODR || data->cntl[5] == HPDDM_KRYLOV_METHOD_BGCRODR) {
      i = HPDDM_ORTHOGONALIZATION_CGS;
      ierr = PetscOptionsEList("-ksp_hpddm_orthogonalization", "Classical (faster) or Modified (more robust) Gram--Schmidt process", "KSPHPDDM", HPDDMOrthogonalization, ALEN(HPDDMOrthogonalization), HPDDMOrthogonalization[HPDDM_ORTHOGONALIZATION_CGS], &i, NULL);CHKERRQ(ierr);
      j = HPDDM_QR_CHOLQR;
      ierr = PetscOptionsEList("-ksp_hpddm_qr", "Distributed QR factorizations computed with Cholesky QR, Classical or Modified Gram--Schmidt process", "KSPHPDDM", HPDDMQR, ALEN(HPDDMQR), HPDDMQR[HPDDM_QR_CHOLQR], &j, NULL);CHKERRQ(ierr);
      data->cntl[2] = static_cast<char>(i) + (static_cast<char>(j) << 2);
      i = PetscMin(30, ksp->max_it - 1);
      ierr = PetscOptionsRangeInt("-ksp_gmres_restart", "Maximum number of Arnoldi vectors generated per cycle", "KSPHPDDM", i, &i, NULL, PetscMin(1, ksp->max_it), PetscMin(ksp->max_it, std::numeric_limits<unsigned short>::max() - 1));CHKERRQ(ierr);
      data->scntl[1] = i;
    }
    if (data->cntl[5] == HPDDM_KRYLOV_METHOD_BCG || data->cntl[5] == HPDDM_KRYLOV_METHOD_BFBCG) {
      j = HPDDM_QR_CHOLQR;
      ierr = PetscOptionsEList("-ksp_hpddm_qr", "Distributed QR factorizations computed with Cholesky QR, Classical or Modified Gram--Schmidt process", "KSPHPDDM", HPDDMQR, ALEN(HPDDMQR), HPDDMQR[HPDDM_QR_CHOLQR], &j, NULL);CHKERRQ(ierr);
      data->cntl[1] = j;
    }
    if (data->cntl[5] == HPDDM_KRYLOV_METHOD_GCRODR || data->cntl[5] == HPDDM_KRYLOV_METHOD_BGCRODR) {
      i = PetscMin(20, data->scntl[1] - 1);
      ierr = PetscOptionsRangeInt("-ksp_hpddm_recycle", "Number of harmonic Ritz vectors to compute", "KSPHPDDM", i, &i, NULL, 1, data->scntl[1] - 1);CHKERRQ(ierr);
      data->icntl[0] = i;
      i = HPDDM_RECYCLE_TARGET_SM;
      ierr = PetscOptionsEList("-ksp_hpddm_recycle_target", "Criterion to select harmonic Ritz vectors", "KSPHPDDM", HPDDMRecycleTarget, ALEN(HPDDMRecycleTarget), HPDDMRecycleTarget[HPDDM_RECYCLE_TARGET_SM], &i, NULL);CHKERRQ(ierr);
      data->cntl[3] = i;
      i = HPDDM_RECYCLE_STRATEGY_A;
      ierr = PetscOptionsEList("-ksp_hpddm_recycle_strategy", "Generalized eigenvalue problem to solve for recycling", "KSPHPDDM", HPDDMRecycleStrategy, ALEN(HPDDMRecycleStrategy), HPDDMRecycleStrategy[HPDDM_RECYCLE_STRATEGY_A], &i, NULL);CHKERRQ(ierr);
      data->cntl[4] = i;
    }
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPView_HPDDM(KSP ksp, PetscViewer viewer)
{
  KSP_HPDDM            *data = (KSP_HPDDM*)ksp->data;
  HPDDM::PETScOperator *op = data->op;
  const PetscScalar    *array = op ? op->storage() : NULL;
  PetscBool            ascii;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &ascii);CHKERRQ(ierr);
  if (op && ascii) {
    ierr = PetscViewerASCIIPrintf(viewer, "HPDDM type: %s\n", HPDDMType[static_cast<PetscInt>(data->cntl[5])]);CHKERRQ(ierr);
    if (data->cntl[5] == HPDDM_KRYLOV_METHOD_GCRODR || data->cntl[5] == HPDDM_KRYLOV_METHOD_BGCRODR) {
      ierr = PetscViewerASCIIPrintf(viewer, "deflation subspace attached? %s\n", PetscBools[array ? PETSC_TRUE : PETSC_FALSE]);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "deflation target: %s\n", HPDDMRecycleTarget[static_cast<PetscInt>(data->cntl[3])]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSetUp_HPDDM(KSP ksp)
{
  KSP_HPDDM      *data = (KSP_HPDDM*)ksp->data;
  Mat            A;
  PetscInt       n, bs;
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPGetOperators(ksp, &A, NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A, &n, NULL);CHKERRQ(ierr);
  ierr = MatGetBlockSize(A, &bs);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompareAny((PetscObject)A, &match, MATSEQBAIJ, MATMPIBAIJ, MATSEQSBAIJ, MATMPISBAIJ, "");CHKERRQ(ierr);
  /* for block formats, the actual size of the underlying arrays are needed */
  if (match) n *= bs;
  ierr = PetscObjectTypeCompareAny((PetscObject)A, &match, MATSEQKAIJ, MATMPIKAIJ, "");CHKERRQ(ierr);
  if (match) n /= bs;
#if defined(PETSC_PKG_HPDDM_VERSION_MAJOR)
#if PETSC_PKG_HPDDM_VERSION_LT(2, 0, 4)
  data->op = new HPDDM::PETScOperator(ksp, n, 1);
#else
  data->op = new HPDDM::PETScOperator(ksp, n);
#endif
#else
  data->op = new HPDDM::PETScOperator(ksp, n, 1);
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPReset_HPDDM(KSP ksp)
{
  KSP_HPDDM *data = (KSP_HPDDM*)ksp->data;
  PetscFunctionBegin;
  if (data->op) {
    delete data->op;
    data->op = NULL;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPDestroy_HPDDM(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPReset_HPDDM(ksp);CHKERRQ(ierr);
  ierr = KSPDestroyDefault(ksp);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp, "KSPHPDDMSetDeflationSpace_C", NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp, "KSPHPDDMGetDeflationSpace_C", NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp, "KSPHPDDMMatSolve_C", NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode KSPSolve_HPDDM_Private(KSP ksp, const PetscScalar *b, PetscScalar *x, PetscInt n)
{
  KSP_HPDDM      *data = (KSP_HPDDM*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = static_cast<PetscInt>(HPDDM::IterativeMethod::solve(*data->op, b, x, n, PetscObjectComm((PetscObject)ksp)));
  /* big assumption from HPDDM: all PetscErrorCode are positive                                            */
  /* if a PETSc call fails inside HPDDM, -ierr is returned (always negative given the previous assumption) */
  /* if a KSPSolve succeeds, the number of iterations is returned instead (always positive or null)        */
  ksp->its = 0;
  if (ierr >= 0) ksp->its = ierr;
  else           return PetscError(PETSC_COMM_SELF, __LINE__, "KSPSolve_HPDDM_Private", __FILE__, -ierr, PETSC_ERROR_INITIAL, "PETSc error detected in HPDDM");
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSolve_HPDDM(KSP ksp)
{
  KSP_HPDDM         *data = (KSP_HPDDM*)ksp->data;
  Mat               A, B;
  PetscScalar       *x, *bt = NULL, **ptr;
  const PetscScalar *b;
  PetscInt          i, j, n;
  PetscBool         flg;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscCitationsRegister(hpddmCitationKSP, &citeKSP);CHKERRQ(ierr);
  ierr = KSPGetOperators(ksp, &A, NULL);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompareAny((PetscObject)A, &flg, MATSEQKAIJ, MATMPIKAIJ, "");CHKERRQ(ierr);
  ierr = VecGetArray(ksp->vec_sol, &x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(ksp->vec_rhs, &b);CHKERRQ(ierr);
  if (!flg) {
    ierr = KSPSolve_HPDDM_Private(ksp, b, x, 1);CHKERRQ(ierr);
  } else {
      ierr = MatKAIJGetScaledIdentity(A, &flg);CHKERRQ(ierr);
      ierr = MatKAIJGetAIJ(A, &B);CHKERRQ(ierr);
      ierr = MatGetBlockSize(A, &n);CHKERRQ(ierr);
      ierr = MatGetLocalSize(B, &i, NULL);CHKERRQ(ierr);
      j = data->op->getDof();
      if (!flg) i *= n; /* S and T are not scaled identities, cannot use block methods */
      if (i != j) { /* switching between block and standard methods */
        delete data->op;
#if defined(PETSC_PKG_HPDDM_VERSION_MAJOR)
#if PETSC_PKG_HPDDM_VERSION_LT(2, 0, 4)
        data->op = new HPDDM::PETScOperator(ksp, i, 1);
#else
        data->op = new HPDDM::PETScOperator(ksp, i);
#endif
#else
        data->op = new HPDDM::PETScOperator(ksp, i, 1);
#endif
      }
      if (flg && n > 1) {
        ierr = PetscMalloc1(i * n, &bt);CHKERRQ(ierr);
        /* from row- to column-major to be consistent with HPDDM */
        HPDDM::Wrapper<PetscScalar>::omatcopy<'T'>(i, n, b, n, bt, i);
        ptr = const_cast<PetscScalar**>(&b);
        std::swap(*ptr, bt);
        HPDDM::Wrapper<PetscScalar>::imatcopy<'T'>(i, n, x, n, i);
      }
      ierr = KSPSolve_HPDDM_Private(ksp, b, x, flg ? n : 1);CHKERRQ(ierr);
      if (flg && n > 1) {
        std::swap(*ptr, bt);
        ierr = PetscFree(bt);CHKERRQ(ierr);
        /* from column- to row-major to be consistent with MatKAIJ format */
        HPDDM::Wrapper<PetscScalar>::imatcopy<'T'>(n, i, x, i, n);
      }
  }
  ierr = VecRestoreArrayRead(ksp->vec_rhs, &b);CHKERRQ(ierr);
  ierr = VecRestoreArray(ksp->vec_sol, &x);CHKERRQ(ierr);
  if (ksp->its < ksp->max_it) ksp->reason = KSP_CONVERGED_RTOL;
  else ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(0);
}

/*@
     KSPHPDDMSetDeflationSpace - Sets the deflation space used by Krylov methods with recycling. This space is viewed as a set of vectors stored in a MATDENSE (column major).

   Input Parameters:
+     ksp - iterative context
-     U - deflation space to be used during KSPSolve()

   Level: intermediate

.seealso:  KSPCreate(), KSPType (for list of available types), KSPHPDDMGetDeflationSpace()
@*/
PetscErrorCode KSPHPDDMSetDeflationSpace(KSP ksp, Mat U)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidHeaderSpecific(U, MAT_CLASSID, 2);
  PetscCheckSameComm(ksp, 1, U, 2);
  ierr = PetscUseMethod(ksp, "KSPHPDDMSetDeflationSpace_C", (KSP, Mat), (ksp, U));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
     KSPHPDDMGetDeflationSpace - Gets the deflation space computed by Krylov methods with recycling or NULL if KSPSolve() has not been called yet. This space is viewed as a set of vectors stored in a MATDENSE (column major). It is the responsibility of the user to free the returned Mat.

   Input Parameter:
.     ksp - iterative context

   Output Parameter:
.     U - deflation space generated during KSPSolve()

   Level: intermediate

.seealso:  KSPCreate(), KSPType (for list of available types), KSPHPDDMSetDeflationSpace()
@*/
PetscErrorCode KSPHPDDMGetDeflationSpace(KSP ksp, Mat *U)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  ierr = PetscUseMethod(ksp, "KSPHPDDMGetDeflationSpace_C", (KSP, Mat*), (ksp, U));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPHPDDMSetDeflationSpace_HPDDM(KSP ksp, Mat U)
{
  KSP_HPDDM            *data = (KSP_HPDDM*)ksp->data;
  HPDDM::PETScOperator *op = data->op;
  Mat                  A;
  const PetscScalar    *array;
  PetscScalar          *copy;
  PetscInt             m1, M1, m2, M2, n2, N2, ldu;
  PetscBool            match;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  if (!op) {
    ierr = KSPSetUp(ksp);CHKERRQ(ierr);
    op = data->op;
  }
  ierr = KSPGetOperators(ksp, &A, NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A, &m1, NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(U, &m2, &n2);CHKERRQ(ierr);
  ierr = MatGetSize(A, &M1, NULL);CHKERRQ(ierr);
  ierr = MatGetSize(U, &M2, &N2);CHKERRQ(ierr);
  if (m1 != m2 || M1 != M2) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Cannot use a deflation space with (m2,M2) = (%D,%D) for a linear system with (m1,M1) = (%D,%D)", m2, M2, m1, M1);
  ierr = PetscObjectTypeCompareAny((PetscObject)U, &match, MATSEQDENSE, MATMPIDENSE, "");CHKERRQ(ierr);
  if (!match) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Provided deflation space not stored in a dense Mat");
  ierr = MatDenseGetArrayRead(U, &array);CHKERRQ(ierr);
  copy = op->allocate(m2, 1, N2);
  if (!copy) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_POINTER, "Memory allocation error");
  ierr = MatDenseGetLDA(U, &ldu);CHKERRQ(ierr);
  HPDDM::Wrapper<PetscScalar>::omatcopy<'N'>(N2, m2, array, ldu, copy, m2);
  ierr = MatDenseRestoreArrayRead(U, &array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPHPDDMGetDeflationSpace_HPDDM(KSP ksp, Mat *U)
{
  KSP_HPDDM            *data = (KSP_HPDDM*)ksp->data;
  HPDDM::PETScOperator *op = data->op;
  Mat                  A;
  const PetscScalar    *array;
  PetscScalar          *copy;
  PetscInt             m1, M1, N2;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  if (!op) {
    ierr = KSPSetUp(ksp);CHKERRQ(ierr);
    op = data->op;
  }
  array = op->storage();
  N2 = op->k();
  if (!array) *U = NULL;
  else {
    ierr = KSPGetOperators(ksp, &A, NULL);CHKERRQ(ierr);
    ierr = MatGetLocalSize(A, &m1, NULL);CHKERRQ(ierr);
    ierr = MatGetSize(A, &M1, NULL);CHKERRQ(ierr);
    ierr = MatCreateDense(PetscObjectComm((PetscObject)ksp), m1, PETSC_DECIDE, M1, N2, NULL, U);CHKERRQ(ierr);
    ierr = MatDenseGetArray(*U, &copy);CHKERRQ(ierr);
    ierr = PetscArraycpy(copy, array, m1 * N2);CHKERRQ(ierr);
    ierr = MatDenseRestoreArray(*U, &copy);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
     KSPHPDDMMatSolve - Solves a linear system with multiple right-hand sides stored as a MATDENSE. Unlike KSPSolve(), B and X must be different matrices.

   Input Parameters:
+     ksp - iterative context
-     B - block of right-hand sides

   Output Parameter:
.     X - block of solutions

   Level: intermediate

.seealso:  KSPSolve(), MatMatSolve(), MATDENSE, PCBJACOBI, PCASM
@*/
PetscErrorCode KSPHPDDMMatSolve(KSP ksp, Mat B, Mat X)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidHeaderSpecific(B, MAT_CLASSID, 2);
  PetscValidHeaderSpecific(X, MAT_CLASSID, 3);
  ierr = PetscUseMethod(ksp, "KSPHPDDMMatSolve_C", (KSP, Mat, Mat), (ksp, B, X));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPViewFinalMatResidual_Internal(KSP ksp, Mat B, Mat X, PetscViewer viewer, PetscViewerFormat format)
{
  Mat            A, R;
  PetscReal      *norms;
  PetscInt       i, N;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCGetOperators(ksp->pc, &A, NULL);CHKERRQ(ierr);
    ierr = MatAssembled(X, &flg);CHKERRQ(ierr);
    if (!flg) {
        ierr = MatSetOption(X, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE);CHKERRQ(ierr);
        ierr = MatAssemblyBegin(X, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(X, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    }
    /* A and X must be assembled */
    ierr = MatMatMult(A, X, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &R);CHKERRQ(ierr);
    ierr = MatAYPX(R, -1.0, B, SAME_NONZERO_PATTERN);
    ierr = MatGetSize(R, NULL, &N);
    ierr = PetscMalloc1(N, &norms);CHKERRQ(ierr);
    ierr = MatGetColumnNorms(R, NORM_2, norms);CHKERRQ(ierr);
    ierr = MatDestroy(&R);CHKERRQ(ierr);
    for (i = 0; i < N; ++i) {
      ierr = PetscViewerASCIIPrintf(viewer, "%s #%D %g\n", i == 0 ? "KSP final norm of residual" : "                          ", i, (double)norms[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(norms);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPHPDDMMatSolve_HPDDM(KSP ksp, Mat B, Mat X)
{
  KSP_HPDDM            *data = (KSP_HPDDM*)ksp->data;
  PC                   pc;
  PC_BJacobi           *bjacobi = NULL;
  PC_ASM               *osm = NULL;
  HPDDM::PETScOperator *op = data->op;
  Mat                  A;
  Vec                  cb, cx;
  const PetscScalar    *b;
  PetscScalar          *x;
  PetscInt             m1, M1, m2, M2, n1, N1, n2, N2, lda;
  PetscBool            match, same_local_solves = PETSC_FALSE;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  if (!op) {
    ierr = KSPSetUp(ksp);CHKERRQ(ierr);
    op = data->op;
  }
  if (B == X) SETERRQ(PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_IDN, "B and X must be different matrices");
  ierr = KSPGetOperators(ksp, &A, NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A, &m1, NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(B, &m2, &n2);CHKERRQ(ierr);
  ierr = MatGetSize(A, &M1, NULL);CHKERRQ(ierr);
  ierr = MatGetSize(B, &M2, &N2);CHKERRQ(ierr);
  if (m1 != m2 || M1 != M2) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Cannot use a block of right-hand sides with (m2,M2) = (%D,%D) for a linear system with (m1,M1) = (%D,%D)", m2, M2, m1, M1);
  ierr = MatGetLocalSize(X, &m1, &n1);CHKERRQ(ierr);
  ierr = MatGetSize(X, &M1, &N1);CHKERRQ(ierr);
  if (m1 != m2 || M1 != M2 || n1 != n2 || N1 != N2) SETERRQ8(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Incompatible block of right-hand sides (m2,M2)x(n2,N2) = (%D,%D)x(%D,%D) and solutions (m1,M1)x(n1,N1) = (%D,%D)x(%D,%D)", m2, M2, n2, N2, m1, M1, n1, N1);
  ierr = PetscObjectTypeCompareAny((PetscObject)B, &match, MATSEQDENSE, MATMPIDENSE, "");CHKERRQ(ierr);
  if (!match) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Provided block of right-hand sides not stored in a dense Mat");
  ierr = PetscObjectTypeCompareAny((PetscObject)X, &match, MATSEQDENSE, MATMPIDENSE, "");CHKERRQ(ierr);
  if (!match) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Provided block of solutions not stored in a dense Mat");
  ierr = MatDenseGetLDA(B, &lda);CHKERRQ(ierr);
  if (m2 != lda) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Unhandled leading dimension lda = %D with m2 = %D", lda, m2);
  ierr = MatDenseGetLDA(X, &lda);CHKERRQ(ierr);
  if (m1 != lda) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Unhandled leading dimension lda = %D with m1 = %D", lda, m1);
  ierr = MatDenseGetArrayRead(B, &b);CHKERRQ(ierr);
  ierr = MatDenseGetArray(X, &x);CHKERRQ(ierr);
  if (N1 > 1) {
    ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
    /* in HPDDM, if BJacobi or ASM is used, a call to PC[BJacobi|ASM]GetSubKSP() is made   */
    /* to know if there is a single subsolver and if it has a MatMatSolve() implementation */
    ierr = PetscObjectTypeCompare((PetscObject)pc, PCBJACOBI, &same_local_solves);CHKERRQ(ierr);
    if (same_local_solves) {
      bjacobi = (PC_BJacobi*)pc->data;
      same_local_solves = bjacobi->same_local_solves;
    }
    if (!bjacobi) {
      ierr = PetscObjectTypeCompare((PetscObject)pc, PCASM, &same_local_solves);CHKERRQ(ierr);
      if (same_local_solves) {
        osm = (PC_ASM*)pc->data;
        same_local_solves = osm->same_local_solves;
      }
    }
    ierr = PetscLogEventBegin(KSP_Solve, ksp, 0, 0, 0);CHKERRQ(ierr);
    ierr = KSPSolve_HPDDM_Private(ksp, b, x, N1);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(KSP_Solve, ksp, 0, 0, 0);CHKERRQ(ierr);
    /* if the PetscBool same_local_solves is not reset after the solve, KSPView() is way too verbose */
    if (same_local_solves) {
      if (bjacobi) bjacobi->same_local_solves = PETSC_TRUE;
      if (osm) osm->same_local_solves = PETSC_TRUE;
    }
    if (ksp->its < ksp->max_it) ksp->reason = KSP_CONVERGED_RTOL;
    else ksp->reason = KSP_DIVERGED_ITS;
    /* stripped-down version of KSPSolve(), which only handles -ksp_view -ksp_converged_reason -ksp_view_final_residual */
    if (ksp->viewReason) {
      ierr = KSPReasonView(ksp, ksp->viewerReason);CHKERRQ(ierr);
    }
    if (ksp->viewFinalRes) {
      ierr = KSPViewFinalMatResidual_Internal(ksp, B, X, ksp->viewerFinalRes, ksp->formatFinalRes);CHKERRQ(ierr);
    }
    if (ksp->view) {
      ierr = KSPView(ksp, ksp->viewer);CHKERRQ(ierr);
    }
  } else {
    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)ksp), 1, m1, M1, b, &cb);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)ksp), 1, m1, M1, x, &cx);CHKERRQ(ierr);
    ierr = KSPSolve(ksp, cb, cx);CHKERRQ(ierr);
    ierr = VecDestroy(&cb);CHKERRQ(ierr);
    ierr = VecDestroy(&cx);CHKERRQ(ierr);
  }
  ierr = MatDenseRestoreArray(X, &x);CHKERRQ(ierr);
  ierr = MatDenseRestoreArrayRead(B, &b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
     KSPHPDDM - Interface with the HPDDM library.

   This KSP may be used to further select methods that are currently not implemented natively in PETSc, e.g., GCRODR [2006], a recycled Krylov method which is similar to KSPLGMRES, see [2016] for a comparison. ex75.c shows how to reproduce the results from the aforementioned paper [2006]. A chronological bibliography of relevant publications linked with KSP available in HPDDM through KSPHPDDM, and not available directly in PETSc, may be found below.

   Options Database Keys:
+   -ksp_richardson_scale <scale, default=1.0> - see KSPRICHARDSON
.   -ksp_gmres_restart <restart, default=40> - see KSPGMRES
.   -ksp_hpddm_type <type, default=gmres> - any of gmres, bgmres, cg, bcg, gcrodr, bgcrodr, or bfbcg
.   -ksp_hpddm_deflation_tol <eps, default=\-1.0> - tolerance when deflating right-hand sides inside block methods (no deflation by default, only relevant with block methods)
.   -ksp_hpddm_enlarge_krylov_subspace <p, default=1> - split the initial right-hand side into multiple vectors (only relevant with nonblock methods)
.   -ksp_hpddm_orthogonalization <type, default=cgs> - any of cgs or mgs, see KSPGMRES
.   -ksp_hpddm_qr <type, default=cholqr> - distributed QR factorizations with any of cholqr, cgs, or mgs (only relevant with block methods)
.   -ksp_hpddm_variant <type, default=left> - any of left, right, or flexible (this option is superseded by KSPSetPCSide())
.   -ksp_hpddm_recycle <n, default=0> - number of harmonic Ritz vectors to compute (only relevant with GCRODR or BGCRODR)
.   -ksp_hpddm_recycle_target <type, default=SM> - criterion to select harmonic Ritz vectors using either SM, LM, SR, LR, SI, or LI (only relevant with GCRODR or BGCRODR)
-   -ksp_hpddm_recycle_strategy <type, default=A> - generalized eigenvalue problem A or B to solve for recycling (only relevant with flexible GCRODR or BGCRODR)

   References:
+   1980 - The Block Conjugate Gradient Algorithm and Related Methods. O'Leary. Linear Algebra and its Applications.
.   2006 - Recycling Krylov Subspaces for Sequences of Linear Systems. Parks, de Sturler, Mackey, Johnson, and Maiti. SIAM Journal on Scientific Computing
.   2013 - A Modified Block Flexible GMRES Method with Deflation at Each Iteration for the Solution of Non-Hermitian Linear Systems with Multiple Right-Hand Sides. Calandra, Gratton, Lago, Vasseur, and Carvalho. SIAM Journal on Scientific Computing.
.   2016 - Block Iterative Methods and Recycling for Improved Scalability of Linear Solvers. Jolivet and Tournier. SC16.
-   2017 - A breakdown-free block conjugate gradient method. Ji and Li. BIT Numerical Mathematics.

   Level: intermediate

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPGMRES, KSPCG, KSPLGMRES, KSPDGMRES
M*/
PETSC_EXTERN PetscErrorCode KSPCreate_HPDDM(KSP ksp)
{
  KSP_HPDDM      *data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(ksp, &data);CHKERRQ(ierr);
  ksp->data = (void*)data;
  ierr = KSPSetSupportedNorm(ksp, KSP_NORM_PRECONDITIONED, PC_LEFT, 2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp, KSP_NORM_UNPRECONDITIONED, PC_RIGHT, 1);CHKERRQ(ierr);
  ksp->ops->setup          = KSPSetUp_HPDDM;
  ksp->ops->solve          = KSPSolve_HPDDM;
  ksp->ops->reset          = KSPReset_HPDDM;
  ksp->ops->destroy        = KSPDestroy_HPDDM;
  ksp->ops->setfromoptions = KSPSetFromOptions_HPDDM;
  ksp->ops->view           = KSPView_HPDDM;
  ierr = PetscObjectComposeFunction((PetscObject)ksp, "KSPHPDDMSetDeflationSpace_C", KSPHPDDMSetDeflationSpace_HPDDM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp, "KSPHPDDMGetDeflationSpace_C", KSPHPDDMGetDeflationSpace_HPDDM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp, "KSPHPDDMMatSolve_C", KSPHPDDMMatSolve_HPDDM);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
