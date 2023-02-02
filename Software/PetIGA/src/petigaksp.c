#include "petiga.h"

PETSC_STATIC_INLINE
PetscBool IGAElementNextFormVector(IGAElement element,IGAFormVector *vec,void **ctx)
{
  IGAForm form = element->parent->form;
  if (!IGAElementNextForm(element,form->visit)) return PETSC_FALSE;
  *vec = form->ops->Vector;
  *ctx = form->ops->VecCtx;
  return PETSC_TRUE;
}

PETSC_STATIC_INLINE
PetscBool IGAElementNextFormMatrix(IGAElement element,IGAFormMatrix *mat,void **ctx)
{
  IGAForm form = element->parent->form;
  if (!IGAElementNextForm(element,form->visit)) return PETSC_FALSE;
  *mat = form->ops->Matrix;
  *ctx = form->ops->MatCtx;
  return PETSC_TRUE;
}

PETSC_STATIC_INLINE
PetscBool IGAElementNextFormSystem(IGAElement element,IGAFormSystem *sys,void **ctx)
{
  IGAForm form = element->parent->form;
  if (!IGAElementNextForm(element,form->visit)) return PETSC_FALSE;
  *sys = form->ops->System;
  *ctx = form->ops->SysCtx;
  return PETSC_TRUE;
}

PETSC_STATIC_INLINE
PetscBool IGAElementNextFormSystemSemiStagger(IGAElement element,IGAFormSystemSemiStagger *sys,void **ctx)
{
  IGAForm form = element->parent->form;
  if (!IGAElementNextForm(element,form->visit)) return PETSC_FALSE;
  *sys = form->ops->SystemSemiStagger;
  *ctx = form->ops->SysCtx;
  return PETSC_TRUE;
}

PETSC_STATIC_INLINE
PetscBool IGAElementNextFormSystemStagger(IGAElement element,IGAElement element_stagger,IGAFormSystemStagger *sys,void **ctx)
{
  IGAForm form = element->parent->form;
  IGAForm form_stagger = element_stagger->parent->form;
  if (!IGAElementNextForm(element,form->visit)) return PETSC_FALSE;
  if (!IGAElementNextForm(element_stagger,form_stagger->visit)) return PETSC_FALSE;
  *sys = form->ops->SystemStagger;
  *ctx = form->ops->SysCtx;
  return PETSC_TRUE;
}

PetscErrorCode IGAComputeVector(IGA iga,Vec vecB)
{
  IGAElement     element;
  IGAPoint       point;
  IGAFormVector  Vector;
  void           *ctx;
  PetscScalar    *B;
  PetscScalar    *F;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(vecB,VEC_CLASSID,2);
  IGACheckSetUp(iga,1);
  IGACheckFormOp(iga,1,Vector);

  ierr = VecZeroEntries(vecB);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(IGA_FormVector,iga,vecB,0,0);CHKERRQ(ierr);

  /* Element loop */
  ierr = IGABeginElement(iga,&element);CHKERRQ(ierr);
  while (IGANextElement(iga,element)) {
    ierr = IGAElementGetWorkVec(element,&B);CHKERRQ(ierr);
    /* FormVector loop */
    while (IGAElementNextFormVector(element,&Vector,&ctx)) {
      /* Quadrature loop */
      ierr = IGAElementBeginPoint(element,&point);CHKERRQ(ierr);
      while (IGAElementNextPoint(element,point)) {
        ierr = IGAPointGetWorkVec(point,&F);CHKERRQ(ierr);
        ierr = Vector(point,F,ctx);CHKERRQ(ierr);
        ierr = IGAPointAddVec(point,F,B);CHKERRQ(ierr);
      }
      ierr = IGAElementEndPoint(element,&point);CHKERRQ(ierr);
    }
    ierr = IGAElementAssembleVec(element,B,vecB);CHKERRQ(ierr);
  }
  ierr = IGAEndElement(iga,&element);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(IGA_FormVector,iga,vecB,0,0);CHKERRQ(ierr);

  ierr = VecAssemblyBegin(vecB);CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (vecB);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGAComputeMatrix(IGA iga,Mat matA)
{
  IGAElement     element;
  IGAPoint       point;
  IGAFormMatrix  Matrix;
  void           *ctx;
  PetscScalar    *A;
  PetscScalar    *K;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(matA,MAT_CLASSID,2);
  IGACheckSetUp(iga,1);
  IGACheckFormOp(iga,1,Matrix);

  ierr = MatZeroEntries(matA);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(IGA_FormMatrix,iga,matA,0,0);CHKERRQ(ierr);

  /* Element loop */
  ierr = IGABeginElement(iga,&element);CHKERRQ(ierr);
  while (IGANextElement(iga,element)) {
    ierr = IGAElementGetWorkMat(element,&A);CHKERRQ(ierr);
    /* FormMatrix loop */
    while (IGAElementNextFormMatrix(element,&Matrix,&ctx)) {
      /* Quadrature loop */
      ierr = IGAElementBeginPoint(element,&point);CHKERRQ(ierr);
      while (IGAElementNextPoint(element,point)) {
        ierr = IGAPointGetWorkMat(point,&K);CHKERRQ(ierr);
        ierr = Matrix(point,K,ctx);CHKERRQ(ierr);
        ierr = IGAPointAddMat(point,K,A);CHKERRQ(ierr);
      }
      ierr = IGAElementEndPoint(element,&point);CHKERRQ(ierr);
    }
    ierr = IGAElementAssembleMat(element,A,matA);CHKERRQ(ierr);
  }
  ierr = IGAEndElement(iga,&element);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(IGA_FormMatrix,iga,matA,0,0);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(matA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (matA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


/*@
   IGAComputeSystem - Form the matrix and vector which represents the
   discretized a(w,u) = L(w).

   Collective on IGA/Mat/Vec

   Input Parameters:
.  iga - the IGA context

   Output Parameters:
+  matA - the matrix obtained from discretization of a(w,u)
-  vecB - the vector obtained from discretization of L(w)

   Notes:
   This routine is used to solve a steady, linear problem. It performs
   matrix/vector assembly standard in FEM. The form provides a routine
   which evaluates the bilinear and linear forms at a point.

   Level: normal

.keywords: IGA, setup linear system, matrix assembly, vector assembly
@*/
PetscErrorCode IGAComputeSystem(IGA iga,Mat matA,Vec vecB)
{
  IGAElement     element;
  IGAPoint       point;
  IGAFormSystem  System;
  void           *ctx;
  PetscScalar    *A,*B;
  PetscScalar    *K,*F;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(matA,MAT_CLASSID,2);
  PetscValidHeaderSpecific(vecB,VEC_CLASSID,3);
  IGACheckSetUp(iga,1);
  IGACheckFormOp(iga,1,System);

  ierr = MatZeroEntries(matA);CHKERRQ(ierr);
  ierr = VecZeroEntries(vecB);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(IGA_FormSystem,iga,matA,vecB,0);CHKERRQ(ierr);

  /* Element loop */
  ierr = IGABeginElement(iga,&element);CHKERRQ(ierr);
  while (IGANextElement(iga,element)) {
    ierr = IGAElementGetWorkMat(element,&A);CHKERRQ(ierr);
    ierr = IGAElementGetWorkVec(element,&B);CHKERRQ(ierr);
    /* FormSystem loop */
    while (IGAElementNextFormSystem(element,&System,&ctx)) {
      /* Quadrature loop */
      ierr = IGAElementBeginPoint(element,&point);CHKERRQ(ierr);
      //PetscPrintf(PETSC_COMM_WORLD,"Point1=%d\n",point->count);
      while (IGAElementNextPoint(element,point)) {
        ierr = IGAPointGetWorkMat(point,&K);CHKERRQ(ierr);
        ierr = IGAPointGetWorkVec(point,&F);CHKERRQ(ierr);
        ierr = System(point,K,F,ctx);CHKERRQ(ierr);
        ierr = IGAPointAddMat(point,K,A);CHKERRQ(ierr);
        ierr = IGAPointAddVec(point,F,B);CHKERRQ(ierr);
      }
      ierr = IGAElementEndPoint(element,&point);CHKERRQ(ierr);
    }
    ierr = IGAElementFixSystem(element,A,B);CHKERRQ(ierr);
    ierr = IGAElementAssembleMat(element,A,matA);CHKERRQ(ierr);
    ierr = IGAElementAssembleVec(element,B,vecB);CHKERRQ(ierr);
  }
  ierr = IGAEndElement(iga,&element);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(IGA_FormSystem,iga,matA,vecB,0);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(matA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (matA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(vecB);CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (vecB);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// IGAComputeSystemStagger is modified by Tianyi Hu, Purdue University
// This assume staggered grid has same dof
PetscErrorCode IGAComputeSystemSemiStagger(IGA iga,Mat matA,Vec vecB,Vec VecUStagger)
{
  IGAElement     element;
  IGAPoint       point;
  const PetscScalar *arrayUStagger;
  Vec               localUStagger;
  IGAFormSystemSemiStagger  System;
  void           *ctx;
  PetscScalar    *A,*B,*UStagger;
  PetscScalar    *K,*F;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(matA,MAT_CLASSID,3);
  PetscValidHeaderSpecific(vecB,VEC_CLASSID,4);
  PetscValidHeaderSpecific(VecUStagger,VEC_CLASSID,5);

  IGACheckSetUp(iga,1);
  IGACheckFormOp(iga,1,System);

  ierr = MatZeroEntries(matA);CHKERRQ(ierr);
  ierr = VecZeroEntries(vecB);CHKERRQ(ierr);
  ierr = IGAGetLocalVecArray(iga,VecUStagger,&localUStagger,&arrayUStagger);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(IGA_FormSystem,iga,matA,vecB,0);CHKERRQ(ierr);

  /* Element loop */
  ierr = IGABeginElement(iga,&element);CHKERRQ(ierr);
  while (IGANextElement(iga,element)) {
    ierr = IGAElementGetWorkMat(element,&A);CHKERRQ(ierr);
    ierr = IGAElementGetWorkVec(element,&B);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayUStagger,&UStagger);CHKERRQ(ierr);
    ierr = IGAElementFixValues(element,UStagger);CHKERRQ(ierr);
    //PetscPrintf(PETSC_COMM_WORLD,"NextElement Loop Running!\n");
    /* FormSystem loop */
    while (IGAElementNextFormSystemSemiStagger(element,&System,&ctx)) {      /* Quadrature loop */
      ierr = IGAElementBeginPoint(element,&point);CHKERRQ(ierr);
      //PetscPrintf(PETSC_COMM_WORLD,"NextFormSystem Loop Running!\n");
      //PetscPrintf(PETSC_COMM_WORLD,"Point1=%d,Point2=%d\n",point->count,point_stagger->count);
      while (IGAElementNextPoint(element,point)) {
        ierr = IGAPointGetWorkMat(point,&K);CHKERRQ(ierr);
        ierr = IGAPointGetWorkVec(point,&F);CHKERRQ(ierr);
        //PetscPrintf(PETSC_COMM_WORLD,"NextPoint Loop Running!\n");
        ierr = System(point,K,F,UStagger,ctx);CHKERRQ(ierr);
        ierr = IGAPointAddMat(point,K,A);CHKERRQ(ierr);
        ierr = IGAPointAddVec(point,F,B);CHKERRQ(ierr);
      }
      ierr = IGAElementEndPoint(element,&point);CHKERRQ(ierr);
    }
    ierr = IGAElementFixSystem(element,A,B);CHKERRQ(ierr);
    ierr = IGAElementAssembleMat(element,A,matA);CHKERRQ(ierr);
    ierr = IGAElementAssembleVec(element,B,vecB);CHKERRQ(ierr);
  }
  ierr = IGAEndElement(iga,&element);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(IGA_FormSystem,iga,matA,vecB,0);CHKERRQ(ierr);

  ierr = IGARestoreLocalVecArray(iga,VecUStagger,&localUStagger,&arrayUStagger);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(matA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (matA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(vecB);CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (vecB);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// IGAComputeSystemStagger is modified by Tianyi Hu, Purdue University
PetscErrorCode IGAComputeSystemStagger(IGA iga,IGA iga_stagger,Mat matA,Vec vecB,Vec VecUStagger)
{
  IGAElement     element,element_stagger;
  IGAPoint       point,point_stagger;
  const PetscScalar *arrayUStagger;
  Vec               localUStagger;
  IGAFormSystemStagger  SystemStagger;
  void           *ctx;
  PetscScalar    *A,*B,*UStagger;
  PetscScalar    *K,*F;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(iga_stagger,IGA_CLASSID,2);
  PetscValidHeaderSpecific(matA,MAT_CLASSID,3);
  PetscValidHeaderSpecific(vecB,VEC_CLASSID,4);
  PetscValidHeaderSpecific(VecUStagger,VEC_CLASSID,5);

  IGACheckSetUp(iga,1);
  IGACheckFormOp(iga,1,System);IGACheckFormOp(iga,1,SystemStagger);

  ierr = MatZeroEntries(matA);CHKERRQ(ierr);
  ierr = VecZeroEntries(vecB);CHKERRQ(ierr);
  ierr = IGAGetLocalVecArray(iga_stagger,VecUStagger,&localUStagger,&arrayUStagger);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(IGA_FormSystem,iga,matA,vecB,0);CHKERRQ(ierr);

  /* Element loop */
  ierr = IGABeginElement(iga,&element);CHKERRQ(ierr);
  ierr = IGABeginElement(iga_stagger,&element_stagger);CHKERRQ(ierr);
  while (IGANextElementStagger(iga,iga_stagger,element,element_stagger)) {
    //PetscPrintf(PETSC_COMM_WORLD,"Elem1=%d,Elem2=%d\n",element->index,element_stagger->index);
    ierr = IGAElementGetWorkMat(element,&A);CHKERRQ(ierr);
    ierr = IGAElementGetWorkVec(element,&B);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element_stagger,arrayUStagger,&UStagger);CHKERRQ(ierr);
    ierr = IGAElementFixValues(element_stagger,UStagger);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"NextElement Loop Running!\n");
    /* FormSystem loop */
    while (IGAElementNextFormSystemStagger(element,element_stagger,&SystemStagger,&ctx)) {
      /* Quadrature loop */
      ierr = IGAElementBeginPoint(element,&point);CHKERRQ(ierr);
      ierr = IGAElementBeginPoint(element_stagger,&point_stagger);CHKERRQ(ierr);
      PetscPrintf(PETSC_COMM_WORLD,"NextFormSystem Loop Running!\n");
      //PetscPrintf(PETSC_COMM_WORLD,"Point1=%d,Point2=%d\n",point->count,point_stagger->count);
      while (IGAElementNextPointStagger(element,element_stagger,point,point_stagger)) {
        ierr = IGAPointGetWorkMat(point,&K);CHKERRQ(ierr);
        ierr = IGAPointGetWorkVec(point,&F);CHKERRQ(ierr);
        PetscPrintf(PETSC_COMM_WORLD,"NextPoint Loop Running!\n");
        ierr = SystemStagger(point,point_stagger,K,F,UStagger,ctx);CHKERRQ(ierr);
        ierr = IGAPointAddMat(point,K,A);CHKERRQ(ierr);
        ierr = IGAPointAddVec(point,F,B);CHKERRQ(ierr);
      }
      ierr = IGAElementEndPoint(element_stagger,&point_stagger);CHKERRQ(ierr);
      ierr = IGAElementEndPoint(element,&point);CHKERRQ(ierr);
    }
    ierr = IGAElementFixSystem(element,A,B);CHKERRQ(ierr);
    ierr = IGAElementAssembleMat(element,A,matA);CHKERRQ(ierr);
    ierr = IGAElementAssembleVec(element,B,vecB);CHKERRQ(ierr);
  }

  ierr = IGAEndElement(iga_stagger,&element_stagger);CHKERRQ(ierr);
  ierr = IGAEndElement(iga,&element);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(IGA_FormSystem,iga,matA,vecB,0);CHKERRQ(ierr);

  ierr = IGARestoreLocalVecArray(iga_stagger,VecUStagger,&localUStagger,&arrayUStagger);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(matA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (matA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(vecB);CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (vecB);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGAKSPFormRHS(KSP ksp,Vec b,void *ctx)
{
  IGA            iga = (IGA)ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidHeaderSpecific(b,VEC_CLASSID,2);
  PetscValidHeaderSpecific(iga,IGA_CLASSID,3);
  if (!iga->form->ops->System) {
    ierr = IGAComputeVector(iga,b);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IGAKSPFormOperators(KSP ksp,Mat A,Mat B,void *ctx)
{
  IGA            iga = (IGA)ctx;
  Vec            rhs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidHeaderSpecific(A,MAT_CLASSID,2);
  PetscValidHeaderSpecific(B,MAT_CLASSID,3);
  PetscValidHeaderSpecific(iga,IGA_CLASSID,4);
  if (!iga->form->ops->System) {
    ierr = IGAComputeMatrix(iga,A);CHKERRQ(ierr);
  } else {
    ierr = KSPGetRhs(ksp,&rhs);CHKERRQ(ierr);
    ierr = IGAComputeSystem(iga,A,rhs);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode KSPSetIGA(KSP ksp,IGA iga)
{
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidHeaderSpecific(iga,IGA_CLASSID,2);
  PetscCheckSameComm(ksp,1,iga,2);
  ierr = PetscObjectCompose((PetscObject)ksp,"IGA",(PetscObject)iga);CHKERRQ(ierr);
  ierr = IGASetOptionsHandlerKSP(ksp);CHKERRQ(ierr);

  ierr = DMIGACreate(iga,&dm);CHKERRQ(ierr);
  ierr = DMKSPSetComputeRHS(dm,IGAKSPFormRHS,iga);CHKERRQ(ierr);
  ierr = DMKSPSetComputeOperators(dm,IGAKSPFormOperators,iga);CHKERRQ(ierr);
  ierr = KSPSetDM(ksp,dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode IGA_OptionsHandler_KSP(PETSC_UNUSED PetscOptionItems *PetscOptionsObject,PetscObject obj,PETSC_UNUSED void *ctx)
{
  KSP            ksp = (KSP)obj;
  DM             dm;
  PetscBool      match,hasmat;
  Mat            mat;
  IGA            iga = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  ierr = PetscObjectQuery((PetscObject)ksp,"IGA",(PetscObject*)&iga);CHKERRQ(ierr);
  ierr = KSPGetDM(ksp,&dm);CHKERRQ(ierr);
  ierr = KSPGetOperatorsSet(ksp,NULL,&hasmat);CHKERRQ(ierr);
  if (!iga && dm) {
    ierr = PetscObjectTypeCompare((PetscObject)dm,DMIGA,&match);CHKERRQ(ierr);
    if (match) {ierr = DMIGAGetIGA(dm,&iga);CHKERRQ(ierr);}
  }
  if (!iga && hasmat) {
    ierr = KSPGetOperators(ksp,NULL,&mat);CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)mat,"IGA",(PetscObject*)&iga);CHKERRQ(ierr);
  }
  if (!iga) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
#if !PETSC_VERSION_LT(3,8,0)
  /* */
  ierr = PetscObjectTypeCompare((PetscObject)ksp,KSPFETIDP,&match);CHKERRQ(ierr);
  if (match) {
    Mat A,P;
    PC  pc;

    ierr = KSPFETIDPGetInnerBDDC(ksp,&pc);CHKERRQ(ierr);
    ierr = KSPGetOperators(ksp,&A,&P);CHKERRQ(ierr);
    ierr = PCSetOperators(pc,A,P);CHKERRQ(ierr);
    ierr = IGAPreparePCBDDC(iga,pc);CHKERRQ(ierr);
  }
  /* */
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode IGA_OptionsHandler_PC(PETSC_UNUSED PetscOptionItems *PetscOptionsObject,PetscObject obj,PETSC_UNUSED void *ctx)
{
  PC             pc = (PC)obj;
  DM             dm;
  PetscBool      match,hasmat;
  Mat            mat;
  IGA            iga = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscObjectQuery((PetscObject)pc,"IGA",(PetscObject*)&iga);CHKERRQ(ierr);
  ierr = PCGetDM(pc,&dm);CHKERRQ(ierr);
  ierr = PCGetOperatorsSet(pc,NULL,&hasmat);CHKERRQ(ierr);
  if (!iga && dm) {
    ierr = PetscObjectTypeCompare((PetscObject)dm,DMIGA,&match);CHKERRQ(ierr);
    if (match) {ierr = DMIGAGetIGA(dm,&iga);CHKERRQ(ierr);}
  }
  if (!iga && hasmat) {
    ierr = PCGetOperators(pc,NULL,&mat);CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)mat,"IGA",(PetscObject*)&iga);CHKERRQ(ierr);
  }
  if (!iga) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  /* */
  ierr = IGAPreparePCMG(iga,pc);CHKERRQ(ierr);
  ierr = IGAPreparePCBDDC(iga,pc);CHKERRQ(ierr);
  /* */
  PetscFunctionReturn(0);
}

static PetscErrorCode OptHdlDel(PETSC_UNUSED PetscObject obj,PETSC_UNUSED void *ctx) {return 0;}

PetscErrorCode IGASetOptionsHandlerKSP(KSP ksp)
{
  PC             pc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  ierr = PetscObjectAddOptionsHandler((PetscObject)ksp,IGA_OptionsHandler_KSP,OptHdlDel,NULL);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = IGASetOptionsHandlerPC(pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGASetOptionsHandlerPC(PC pc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscObjectAddOptionsHandler((PetscObject)pc,IGA_OptionsHandler_PC,OptHdlDel,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   IGACreateKSP - Creates a KSP (linear solver) which uses the same
   communicators as the IGA.

   Logically collective on IGA

   Input Parameter:
.  iga - the IGA context

   Output Parameter:
.  ksp - the KSP

   Level: normal

.keywords: IGA, create, KSP
@*/
PetscErrorCode IGACreateKSP(IGA iga,KSP *ksp)
{
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(ksp,2);

  ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
  ierr = KSPCreate(comm,ksp);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)*ksp,"IGA",(PetscObject)iga);CHKERRQ(ierr);
  ierr = IGASetOptionsHandlerKSP(*ksp);CHKERRQ(ierr);

  /*ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);*/
  /*ierr = KSPSetOperators(*ksp,A,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);*/
  /*ierr = MatDestroy(&A);CHKERRQ(ierr);*/
  PetscFunctionReturn(0);
}
