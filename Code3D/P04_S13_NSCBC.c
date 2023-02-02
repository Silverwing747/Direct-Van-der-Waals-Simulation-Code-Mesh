#include "P04_S13_header.h"

LocalStruct St;

int main(int argc, char *argv[]) {

  //***************************Initialized PETSC***************************//
  PetscErrorCode  ierr;
  PetscMPIInt     rank,size;
  AppCtx          user;
  PetscInt        i,j;
  char            Version[PETSC_MAX_PATH_LEN];

  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);
  ierr = PetscGetVersion(Version,sizeof(Version));CHKERRQ(ierr);
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  PetscPrintf(PETSC_COMM_WORLD,"----------PESTC and System Info----------\n");
  PetscPrintf(PETSC_COMM_WORLD,"PETSC Version is:       %s\n",Version);
  PetscPrintf(PETSC_COMM_WORLD,"Additional Notes:       This PETSC is modified by Tianyi Hu (Purdue University):\n");
  PetscPrintf(PETSC_COMM_WORLD,"                         - New adaptive time scheme is implemented\n");
  PetscPrintf(PETSC_COMM_WORLD,"                         - Time derivative is printed to restart simulation\n");
  PetscPrintf(PETSC_COMM_WORLD,"Processors Info:        size = %d, rank = %d \n",size,rank);
  PrintInfo();

  //PetscPrintf(PETSC_COMM_WORLD,"Test:        Nan = %d, Real = %d \n",PetscIsNanReal(0.0/0.0),PetscIsNanReal(1.0));
  //return 0;

  //***************************User input section***************************//
  // Model, mesh and basis
  user.dim = DIM; // Problem dimension
  user.N = 128; // Mesh size per unit length
  user.p = 2; // basis order
  user.Coord = 0; // Coordinate, 0=Cartisian, 1=Cylindrical, 2=Spherical 
  user.TIM = 0; // 0 = original NSK, 1 = TIM, 2 = Improved TIM
  user.d_v = 1e-2; user.d_l = 1e-2; // Parameter for modified EoS for Improved TIM, should be a small number
  user.Energy = 0; // 0: Isothermo, 1: full energy
  user.Sponge = 0; // 0 = no sponge zone, 1: with sponge
  // Notice that sometimes smaller rho_TSAlpha will lead to slower convergence

  // SUPG and DC
  user.SUPG = 3; // 0: No SUPG, 1: implicit fast SUPG, 2: explicit fast SUPG, 3: full SUPG, 4: explicit full SUPG
  user.DBIter = 10; // Denman Beavers max iteration #
  user.DC = 2; // 0: No DC, 1: Explicit DC1, 2: Explicit DC2
  user.LocalScale = 1; // 0: Scale = 1, 1: Scale small at liquid phase
  user.NSK_Mod = 1; // 0: Local Stabilization, 1: non-local Stabilization
  user.C_SUPG = 0.1; // stabilization parameter for SUPG
  user.C_DC = 0.1;
  user.alpha_DC = 0.001;

  // Model parameter
  user.Re = INFINITY;// Re = 1.5e+6 corresponds to 1e-3 m length scale
  user.mu_ratio = 100.0;// mu_liquid / mu_vapor
  user.Ca = 1.0/2000.0; 
  user.Pe = INFINITY;//10000.0;
  user.Ge = 0; // Gravity coefficient
  user.theta = 0.4624; St.T = user.theta;// Temperature for isothermo case, theta=0.4624 is 300 K
  user.delta = 10.0; // For air, delta = 2.5
  user.F = 1.0;

  // Problem set up and BC
  user.Case = 0; // 0:other cases, 1: Interface oscillation
  user.BCType = 4; // 1 = periodic, 2 = all wall, 3 = all slip wall, 4+ other special BC (detail see description)
  user.ICScale_v = 1.0; 
  user.ICScale_l = 1.0; // liquid scale 1.000012 results 1 atm liquid, for TIM2 it's 1.000343 when d=0.01 1.0000425 when d=1e-4
  user.IC_Scale = 0.5;
  user.u_BC[0] = 0.0; user.u_BC[1] = 0.0; user.u_BC[2] = 0.0;
  // Initial Condition
  user.R1 = -INFINITY; user.C1[0] = 0.5; user.C1[1] = 0.5; user.C1[2] = 0.25;
  user.R2 = -INFINITY; user.C2[0] = 0.25; user.C2[1] = 0.50; user.C2[2] = 0.50;
  user.R3 = -INFINITY; user.C3[0] = 0.40; user.C3[1] = 0.75; user.C3[2] = 0.70;

  // Sponge BC
  user.SpongeCoef = 25.0;
  user.SpongeZone[0][0] = 0.2; // x
  user.SpongeZone[0][1] = 0.2; // x
  user.SpongeZone[1][0] = 0.5; // y
  user.SpongeZone[1][1] = 0.5; // y
  user.SpongeZone[2][0] = 0.0; // z
  user.SpongeZone[2][1] = 0.0; // z
  user.Penalty_CoM = 1e+4;
  user.Penalty_CoLM = 1e+4;
  user.K_outflow = 0.0; // 0:non-reflecting

  // Define output and monitor options
  user.PrintTimeInterval = 0.01;
  PetscBool output = PETSC_TRUE, monitor = PETSC_TRUE;

  // Define initial file name
  char      der_initial[PETSC_MAX_PATH_LEN] = "Der_";
  char      initial[PETSC_MAX_PATH_LEN] = "NSK_";
  char      desiredTime[PETSC_MAX_PATH_LEN] = "IC";
  char      dataType[PETSC_MAX_PATH_LEN] = ".dat";

  // Set discretization options
  PetscInt ne;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "NSK Options", "IGA");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-p", "polynomial order", __FILE__, user.p, &user.p, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-C", "global continuity order", __FILE__, user.C, &user.C, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-N", "global continuity order", __FILE__, user.N, &user.N, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-BC", "Boundary condition", __FILE__, user.BCType, &user.BCType, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "problem dimension", __FILE__, user.dim, &user.dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-TIM", "TIM", __FILE__, user.TIM, &user.TIM, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-BCType", "BCType", __FILE__, user.BCType, &user.BCType, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-Case", "Case", __FILE__, user.Case, &user.Case, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-SUPG", "Model", __FILE__, user.SUPG, &user.SUPG, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-DC", "Model", __FILE__, user.DC, &user.DC, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-Sponge","Enable output files",__FILE__,user.Sponge,&user.Sponge,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-LocalScale", "Model", __FILE__, user.LocalScale, &user.LocalScale, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-Energy","Enable output files",__FILE__,user.Energy,&user.Energy,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-PrintTimeInterval", "Time interval of two data file output", __FILE__, user.PrintTimeInterval, &user.PrintTimeInterval, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-Re", "Model", __FILE__, user.Re, &user.Re, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-Ca", "Model", __FILE__, user.Ca, &user.Ca, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-Pe", "Model", __FILE__, user.Pe, &user.Pe, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-Theta", "Model", __FILE__, user.theta, &user.theta, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-Delta", "Model", __FILE__, user.delta, &user.delta, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-C_SUPG", "Model", __FILE__, user.C_SUPG, &user.C_SUPG, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-C_DC", "Model", __FILE__, user.C_DC, &user.C_DC, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-NSK_Mod", "Model", __FILE__, user.NSK_Mod, &user.NSK_Mod, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-F", "Scaling Coeff", __FILE__, user.F, &user.F, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-Coord", "Type of coordinate", __FILE__, user.Coord, &user.Coord, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ICScale_v", "Model", __FILE__, user.ICScale_v, &user.ICScale_v, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ICScale_l", "Model", __FILE__, user.ICScale_l, &user.ICScale_l, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ICScale", "Model", __FILE__, user.IC_Scale, &user.IC_Scale, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-R1", "Model", __FILE__, user.R1, &user.R1, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-R2", "Model", __FILE__, user.R2, &user.R2, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-R3", "Model", __FILE__, user.R3, &user.R3, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-SpongeCoef", "SpongeCoef", __FILE__, user.SpongeCoef, &user.SpongeCoef, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-K_outflow", "K_outflow", __FILE__, user.K_outflow, &user.K_outflow, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-IC","Initial condition file time",__FILE__,desiredTime,desiredTime,256,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsRealArray("-u_BC", "u_BC",__FILE__,&user.u_BC[0],(ne=3,&ne),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsRealArray("-SpongeZone", "SpongeZone",__FILE__,&user.SpongeZone[0][0],(ne=6,&ne),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsRealArray("-DomainSize", "DomainSize",__FILE__,&user.DomainSize[0][0],(ne=6,&ne),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsRealArray("-C1", "C1",__FILE__,&user.C1[0],(ne=3,&ne),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsRealArray("-C2", "C2",__FILE__,&user.C2[0],(ne=3,&ne),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsRealArray("-C3", "C3",__FILE__,&user.C3[0],(ne=3,&ne),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  strcat(initial,desiredTime); strcat(initial,dataType); strcat(der_initial,initial);

  // Boundary Condition
  St.T = user.theta;
  CalcSaturation(&user,user.theta,&St.rho_l,&St.rho_v,&St.rho_m,&St.e_sat_l,&St.e_sat_v);
  user.rho_v_IC = user.ICScale_v * St.rho_v; user.rho_l_IC = user.ICScale_l * St.rho_l;
  user.rho_inlet = user.rho_l_IC; user.rho_outlet = user.rho_l_IC;
  user.T_inlet = user.theta; user.T_outlet = user.theta;
  CalcValue(&user,1,user.rho_inlet,user.T_inlet,NULL);
  St.rho = user.rho_inlet; TIMModification(&user);
  user.p_BC = St.p;
  
  // Interfacial related parameters
  if (!user.TIM) user.F = 1.0;
  user.Ca2 = SQ(user.Ca) * user.F;
  if (user.NSK_Mod == 0) user.Ca2_SUPG = 0.0;
  else user.Ca2_SUPG = user.Ca2;

  user.u_IC[0] = user.u_BC[0]; user.u_IC[1] = user.u_BC[1]; user.u_IC[2] = user.u_BC[2]; 
  user.T_IC = user.T_inlet;

  // Define Domain, Mesh and Basis 
  user.dof = user.dim + user.Energy + 2; user.C = user.p - 1;
  if(user.Case == 1){
    user.DomainSize[0][0] = 0.0; user.DomainSize[0][1] = 0.005 * user.F; user.MeshSize[0] = user.N;//user.N * (user.DomainSize[0][1] - user.DomainSize[0][0]);// X-axis
    user.DomainSize[1][0] = -0.005 * user.F; user.DomainSize[1][1] = 0.005 * user.F; user.MeshSize[1] = user.N*2;//user.N * (user.DomainSize[1][1] - user.DomainSize[1][0]);// Y-axis
    user.DomainSize[2][0] = 0.0; user.DomainSize[2][1] = 1.0; user.MeshSize[2] = user.N * (user.DomainSize[2][1] - user.DomainSize[2][0]);// Z-axis
  }else{
    user.DomainSize[0][0] = 0.0; user.DomainSize[0][1] = 1.0; user.MeshSize[0] = user.N * (user.DomainSize[0][1] - user.DomainSize[0][0]);// X-axis
    user.DomainSize[1][0] = 0.0; user.DomainSize[1][1] = 1.0; user.MeshSize[1] = user.N * (user.DomainSize[1][1] - user.DomainSize[1][0]);// Y-axis
    user.DomainSize[2][0] = 0.0; user.DomainSize[2][1] = 1.0; user.MeshSize[2] = user.N * (user.DomainSize[2][1] - user.DomainSize[2][0]);// Z-axis
  }


  //***************************Monitor Parameters***************************//
  //***************************Monitor Parameters***************************//
  PetscPrintf(PETSC_COMM_WORLD,"----------Problem Setup----------\n");
  PetscPrintf(PETSC_COMM_WORLD,"Case:                   %d (0:other cases, 1: Interface oscillation)\n",user.Case);
  PetscPrintf(PETSC_COMM_WORLD,"vapor_IC_scale:         %.6f\n",user.ICScale_v);
  PetscPrintf(PETSC_COMM_WORLD,"liquid_IC_scale:        %.6f\n",user.ICScale_l);
  PetscPrintf(PETSC_COMM_WORLD,"Velocity BC (mag):      %.6f\n",sqrt(SQ(user.u_BC[0])+SQ(user.u_BC[1])+SQ(user.u_BC[2])));
  PetscPrintf(PETSC_COMM_WORLD,"            x-dir:      %.6f\n",user.u_BC[0]);
  PetscPrintf(PETSC_COMM_WORLD,"            y-dir:      %.6f\n",user.u_BC[1]);
  PetscPrintf(PETSC_COMM_WORLD,"            z-dir:      %.6f\n",user.u_BC[2]);
  PetscPrintf(PETSC_COMM_WORLD,"----------Model Info----------\n");
  PetscPrintf(PETSC_COMM_WORLD,"Dim:                    %d\n",user.dim);
  PetscPrintf(PETSC_COMM_WORLD,"TIM:                    %d (0=Original NSK, 1=TIM, 2=Improved TIM)\n",user.TIM);
  PetscPrintf(PETSC_COMM_WORLD,"                        d_l = %.2e, d_v = %.2e (Parameter for modified EoS, used in Improved TIM)\n",user.d_l,user.d_v);
  PetscPrintf(PETSC_COMM_WORLD,"Coord:                  %.0f (0=Cartisian, 1=Cylindrical, 2=Spherical)\n",user.Coord);
  PetscPrintf(PETSC_COMM_WORLD,"Energy:                 %d (0=Isothermo, 1=Full Energy)\n",user.Energy);
  PetscPrintf(PETSC_COMM_WORLD,"SUPG:                   %d (0: No SUPG, 1: Fast SUPG, 2: Fast Explicit SUPG, 3: Full SUPG, 4: Full SUPG Explicit)\n",user.SUPG);
  PetscPrintf(PETSC_COMM_WORLD," - DB It                %d (Max number of Denmon Beaver iteration)\n",user.DBIter);
  PetscPrintf(PETSC_COMM_WORLD," - C_SUPG               %.6f\n",user.C_SUPG);
  PetscPrintf(PETSC_COMM_WORLD,"DC:                     %d (0: No DC, 1: Explicit DC1, 2: Explicit DC2)\n",user.DC);
  PetscPrintf(PETSC_COMM_WORLD," - C_DC                 %.6f\n",user.C_DC); 
  PetscPrintf(PETSC_COMM_WORLD," - LocalScale           %d\n",user.LocalScale);  
  PetscPrintf(PETSC_COMM_WORLD,"NSK_Mod                 %.0f (0: Local; 1: Non-local; 2: Truncate dp_drho)\n",user.NSK_Mod);
  PetscPrintf(PETSC_COMM_WORLD,"----------Physical Parameters----------\n");
  PetscPrintf(PETSC_COMM_WORLD,"delta                   %.6f (cv/R)\n",user.delta);
  PetscPrintf(PETSC_COMM_WORLD,"theta                   %.6f\n",user.theta);
  PetscPrintf(PETSC_COMM_WORLD,"                        p_sat = %.6e (Pa), rho_l = %.6e, rho_v = %.6e, rho_m = %.6e\n",St.p_sat * 1.6669e+09,St.rho_l,St.rho_v,St.rho_m);
  PetscPrintf(PETSC_COMM_WORLD,"                        dp_drho_v = %.6e, dp_drho_l = %.6e, ratio = %.6e\n",St.dp_drho_v,St.dp_drho_l,St.dp_drho_v/St.dp_drho_l);  
  PetscPrintf(PETSC_COMM_WORLD,"Ca:                     %.6e\n",user.Ca);
  PetscPrintf(PETSC_COMM_WORLD,"Cavitation Number:      %.6f \n",(user.p_BC - St.p_sat) / (0.5 * user.rho_inlet * SQ(user.u_BC[0])));
  PetscPrintf(PETSC_COMM_WORLD,"Re:                     %.6e \n",user.Re);
  PetscPrintf(PETSC_COMM_WORLD,"mu_l/mu_v:              %.6e \n",user.mu_ratio);
  PetscPrintf(PETSC_COMM_WORLD,"Re_inf:                 %.6e (Assume L_c = 1.0, for cylinder L_c = 2R, for wedge L_c = h, double check with geometry file!)\n",user.Re * user.rho_inlet * user.u_BC[0]);
  PetscPrintf(PETSC_COMM_WORLD,"Pe:                     %.6e (rPe = %.6e)\n",user.Pe,user.rPe);
  PetscPrintf(PETSC_COMM_WORLD,"Ge:                     %.6e\n",user.Ge);
  PetscPrintf(PETSC_COMM_WORLD,"F:                      %.6e\n",user.F);
  PetscPrintf(PETSC_COMM_WORLD,"----------Boundary Parameter----------\n");
  PetscPrintf(PETSC_COMM_WORLD,"p_BC:                   %.6e (Pa)\n",user.p_BC * 1.6669e+09);
  PetscPrintf(PETSC_COMM_WORLD,"K_outflow:              %.3f \n",user.K_outflow);
  PetscPrintf(PETSC_COMM_WORLD,"Penalty:                %.2f (Inlet)\n",user.Penalty_CoM);
  PetscPrintf(PETSC_COMM_WORLD,"                        %.2f (Outlet)\n",user.Penalty_CoLM);
  PetscPrintf(PETSC_COMM_WORLD,"----------Sponge Zone----------\n");  
  PetscPrintf(PETSC_COMM_WORLD,"Sponge On/Off:          %d (0 = off; 1 = on)\n",user.Sponge);
  PetscPrintf(PETSC_COMM_WORLD,"Sponge Coef:            %.6f\n",user.SpongeCoef);
  PetscPrintf(PETSC_COMM_WORLD,"      x-dir:            %.6f %.6f\n",user.SpongeZone[0][0],user.SpongeZone[0][1]);
  PetscPrintf(PETSC_COMM_WORLD,"      y-dir:            %.6f %.6f\n",user.SpongeZone[1][0],user.SpongeZone[1][1]);
  PetscPrintf(PETSC_COMM_WORLD,"      z-dir:            %.6f %.6f\n",user.SpongeZone[2][0],user.SpongeZone[2][1]);
  //***************************Set up IGA, IGA_Axis and Boundary Condition***************************//
  // Set up IGA
  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDim(iga,user.dim);CHKERRQ(ierr);
  ierr = IGASetDof(iga,user.dof);CHKERRQ(ierr);
  ierr = IGASetFieldName(iga,0,"rho"); CHKERRQ(ierr);
  if (user.Energy) ierr = IGASetFieldName(iga,user.dim+1,"T"); CHKERRQ(ierr);
  ierr = IGASetFieldName(iga,user.dof-1,"aux"); CHKERRQ(ierr);
  for(i=0;i<user.dim;i++) ierr = IGASetFieldName(iga,i+1,"u_vec"); CHKERRQ(ierr);

  // Set up IGA_Axis
  PetscPrintf(PETSC_COMM_WORLD,"----------Boundary Condition----------\n");
  PetscPrintf(PETSC_COMM_WORLD,"Boundary Type:          %d (1 = periodic,2 = all wall,3 = all slip wall,4+ other special BC)\n",user.BCType);
  for(i=0;i<3;i++){for(j=0;j<2;j++) user.BCType_Identify[i][j] = 0;} // Initialize BCType_Identify
  IGAAxis axis[user.dim];
  for(i=0;i<user.dim;i++){
    ierr = IGAGetAxis(iga,i,&axis[i]);CHKERRQ(ierr);
    switch(user.BCType){
      case 1: // Periodic
        PetscPrintf(PETSC_COMM_WORLD,"Periodic BC in all direction - Validated\n");
        ierr = IGAAxisSetPeriodic(axis[i],PETSC_TRUE);CHKERRQ(ierr); // Must do this before set degree!!!!!!
        break;
      case 2: // All Wall
        PetscPrintf(PETSC_COMM_WORLD,"No-slip wall BC in all direction - Validated\n");
        for (j=0;j<user.dim;j++){
          ierr = IGASetBoundaryValue(iga,i,0,j+1,0.0);CHKERRQ(ierr); ierr = IGASetBoundaryValue(iga,i,1,j+1,0.0);CHKERRQ(ierr);
          user.BCType_Identify[i][0] = 3; user.BCType_Identify[i][1] = 3;
        }
        break;
      case 3: // Slip wall BC
        ierr = IGASetBoundaryValue(iga,i,0,i+1,0.0); ierr = IGASetBoundaryValue(iga,i,1,i+1,0.0);CHKERRQ(ierr);
        PetscPrintf(PETSC_COMM_WORLD,"Slip wall BC in all direction - Validated\n");
        break;
      case 4:
        PetscPrintf(PETSC_COMM_WORLD,"Top and botton free slip wall, all other direction periodic - Validated\n");
        if (i == 1){
          ierr = IGASetBoundaryValue(iga,i,0,i+1,0.0);
          ierr = IGASetBoundaryValue(iga,i,1,i+1,0.0);
        }
        else{ ierr = IGAAxisSetPeriodic(axis[i],PETSC_TRUE);CHKERRQ(ierr);}
        break;
      case 5:
        PetscPrintf(PETSC_COMM_WORLD,"Bottom no-slip wall, top free slip wall, left velocity inlet, right pressure outlet, z direction periodic - Validated\n");
        if (i == 0){ // X axis
          user.BCType_Identify[i][1] = 0; // outflow
          for (j=0;j<user.dim;j++) ierr = IGASetBoundaryValue(iga,i,1,j+1,user.u_BC[j]);CHKERRQ(ierr); // outflow 
          ierr = IGASetBoundaryValue(iga,i,1,0,user.rho_inlet);CHKERRQ(ierr);

          for (j=0;j<user.dim;j++) ierr = IGASetBoundaryValue(iga,i,0,j+1,user.u_BC[j]);CHKERRQ(ierr); // Velocity inlet
          user.BCType_Identify[i][0] = 0; // Velocity inlet
        }
        if (i == 1){ // Y axis
          ierr = IGASetBoundaryValue(iga,i,1,i+1,0.0);CHKERRQ(ierr);
          for (j=0;j<user.dim;j++) ierr = IGASetBoundaryValue(iga,i,0,j+1,0.0);CHKERRQ(ierr); // Bottom no split
          user.BCType_Identify[i][0] = 3; 
        }
        if (i == 2) ierr = IGAAxisSetPeriodic(axis[i],PETSC_TRUE);CHKERRQ(ierr); // Z axis: periodic
        break;
      case 6:
        PetscPrintf(PETSC_COMM_WORLD,"Left symmetry, bottom no slip, top and right sponge, z direction periodic - Validated\n");
        if (i == 0){
          ierr = IGASetBoundaryValue(iga,i,0,i+1,0.0);
          user.BCType_Identify[i][1] = 2; 
          if(user.Energy) ierr = IGASetBoundaryValue(iga,i,1,DIM+1,user.T_inlet);
          }
        if (i == 1){ // Y axis
          for (j=0;j<user.dim;j++) ierr = IGASetBoundaryValue(iga,i,0,j+1,0.0);CHKERRQ(ierr); // Bottom no slip
          user.BCType_Identify[i][0] = 3;
          //ierr = IGASetBoundaryValue(iga,i,1,i+1,0.0);
          user.BCType_Identify[i][1] = 2; 
          if(user.Energy) ierr = IGASetBoundaryValue(iga,i,1,DIM+1,user.T_inlet);
        }
        if (i == 2) ierr = IGAAxisSetPeriodic(axis[i],PETSC_TRUE);CHKERRQ(ierr); // Z axis: periodic
        break;
      case 7:
        PetscPrintf(PETSC_COMM_WORLD,"Left symmetric BC, right all pressure outlet (quarter domain)\n");
        ierr = IGASetBoundaryValue(iga,i,0,i+1,0.0);
        if(user.Energy) ierr = IGASetBoundaryValue(iga,i,1,DIM+1,user.T_inlet);
        user.BCType_Identify[i][1] = 2; 
        break;
      case 8:
        PetscPrintf(PETSC_COMM_WORLD," Flow over cylinder, z direction periodic (must pair up with sponge zone BC!)\n");
        if (i == 1){ 
          for (j=0;j<user.dim;j++) ierr = IGASetBoundaryValue(iga,i,0,j+1,0.0);CHKERRQ(ierr); // Interior cylinder no slip
          ierr = IGASetBoundaryValue(iga,i,1,0,user.rho_inlet);CHKERRQ(ierr);
          if(user.Energy) ierr = IGASetBoundaryValue(iga,i,1,user.dim+1,user.T_inlet);CHKERRQ(ierr);
          ierr = IGASetBoundaryValue(iga,i,1,user.dof-1,0.0); 
          for (j=0;j<user.dim;j++) ierr = IGASetBoundaryValue(iga,i,1,j+1,user.u_BC[j]);CHKERRQ(ierr);
        }
        else ierr = IGAAxisSetPeriodic(axis[i],PETSC_TRUE);CHKERRQ(ierr); // Z axis: periodic
        break;
      case 9:
        PetscPrintf(PETSC_COMM_WORLD,"All pressure outlet (non-reflecting)\n");
        //if (i == 0) {ierr = IGAAxisSetPeriodic(axis[i],PETSC_TRUE);CHKERRQ(ierr);} // Z axis: periodic
        //else{
          user.BCType_Identify[i][0] = 2; 
          user.BCType_Identify[i][1] = 2; 
          ierr = IGASetBoundaryValue(iga,i,0,user.dof-1,0.0); 
          ierr = IGASetBoundaryValue(iga,i,1,user.dof-1,0.0); 

          for (j=0;j<user.dim;j++) ierr = IGASetBoundaryValue(iga,i,0,j+1,user.u_BC[j]);CHKERRQ(ierr); // Velocity inlet
          for (j=0;j<user.dim;j++) ierr = IGASetBoundaryValue(iga,i,1,j+1,user.u_BC[j]);CHKERRQ(ierr); // Velocity inlet
          user.BCType_Identify[i][0] = 1; // Velocity inlet
          user.BCType_Identify[i][1] = 1; // Velocity inlet
        //}
        break; 
      case 10:
        PetscPrintf(PETSC_COMM_WORLD,"Left and right symmetry, top and bottom pressure outflow\n");
        if (i == 1){
          user.BCType_Identify[i][0] = 2; 
          user.BCType_Identify[i][1] = 2; 
          ierr = IGASetBoundaryValue(iga,i,0,user.dof-1,0.0); 
          ierr = IGASetBoundaryValue(iga,i,1,user.dof-1,0.0); 
        }else{
          ierr = IGASetBoundaryValue(iga,i,0,i+1,0.0); 
          ierr = IGASetBoundaryValue(iga,i,1,i+1,0.0);
          //ierr = IGAAxisSetPeriodic(axis[i],PETSC_TRUE);CHKERRQ(ierr);
        }
        
        break;
    }
    ierr = IGAAxisSetDegree(axis[i],user.p);CHKERRQ(ierr);
    ierr = IGAAxisInitUniform(axis[i],user.MeshSize[i],user.DomainSize[i][0],user.DomainSize[i][1],user.C);CHKERRQ(ierr);
  }

  PetscPrintf(PETSC_COMM_WORLD,"                        Boundary Integral Flag:\n");
  PetscPrintf(PETSC_COMM_WORLD,"                        0=no boundary integral, 1=inflow, 2=outflow, 3=no-slip wall\n");
  PetscPrintf(PETSC_COMM_WORLD,"                        For NSK, you need use set laplace rho = 0 in addition to characteristic BC\n");
  for(i=0;i<3;i++){PetscPrintf(PETSC_COMM_WORLD,"                        - i = %d:",i);
    for(j=0;j<2;j++){PetscPrintf(PETSC_COMM_WORLD," %d",user.BCType_Identify[i][j]);
      if(user.BCType_Identify[i][j]){IGASetBoundaryForm(iga,i,j,PETSC_TRUE);CHKERRQ(ierr);}
    } PetscPrintf(PETSC_COMM_WORLD,"\n");
  }

  ierr = IGASetFormIEFunction(iga,Residual,&user);CHKERRQ(ierr);
  ierr = IGASetFormIEJacobian(iga,Jacobian,&user);CHKERRQ(ierr);

  PetscBool fd = IGAGetOptBool(NULL,"-fd",PETSC_FALSE);
  if (fd) {
    PetscPrintf(PETSC_COMM_WORLD,"FDJacobian!\n");
    ierr = IGASetFormIEJacobian(iga,IGAFormIEJacobianFD,&user);CHKERRQ(ierr);
  }

  PetscPrintf(PETSC_COMM_WORLD,"----------Option -iga_view----------\n");

  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);
  user.iga = iga;

  //***************************Set up TS and adaptive options***************************//
  // Set up TS
  TS ts;
  ierr = IGACreateTS(iga,&ts);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,10000.0);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,0.001);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSALPHA);CHKERRQ(ierr);
  ierr = TSAlphaSetRadius(ts,0.5);CHKERRQ(ierr);

  // Set up TS Adapt time step
  ts->TianyiAdaptTime1 = 1; ts->TianyiAdaptTime2 = 1; // 1 is true, 0 is false
  ts->TianyiAdjustFactor1 = 0.75; ts->TianyiAdjustFactor2 = 0.05;
  ts->min_nits = 3; ts->min_lits = 300;
  ts->max_nits = 4; ts->max_lits = 400;
  ts->max_reject = 10;
  ierr = TSSetMaxSNESFailures(ts,-1);CHKERRQ(ierr);
  ierr = TSAlphaUseAdapt(ts,PETSC_FALSE);CHKERRQ(ierr);

  // Set up monitor and output options
  if (monitor) {ierr = TSMonitorSet(ts,NSKMonitor,&user,NULL);CHKERRQ(ierr);}
  if (output)  {ierr = TSMonitorSet(ts,OutputMonitor,&user,NULL);CHKERRQ(ierr);}
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  user.ts = ts;

  // Convergence Test
  SNES nonlin;
  ierr = TSGetSNES(ts,&nonlin);CHKERRQ(ierr);
  ierr = SNESSetConvergenceTest(nonlin,SNESDOFConvergence,&user,NULL);CHKERRQ(ierr);
  user.nonlin = nonlin;

  // TS Print Check
  PetscPrintf(PETSC_COMM_WORLD,"----------TS Setup----------\n");
  PetscPrintf(PETSC_COMM_WORLD,"Basic Setup Info:       dt = %f, t_max = %f\n",ts->time_step,ts->max_time);
  PetscPrintf(PETSC_COMM_WORLD,"                        Monitor = %s, Output = %s\n",monitor ? "TRUE":"FALSE",output ? "TRUE":"FALSE");
  PetscPrintf(PETSC_COMM_WORLD,"                        Print Time Interval = %f\n",user.PrintTimeInterval);
  PetscPrintf(PETSC_COMM_WORLD,"TS Adapt Time Setup:    PETSCAdaptTime = %s, Tpye = BASIC\n",PETSC_FALSE ? "TRUE":"FALSE");
  PetscPrintf(PETSC_COMM_WORLD,"                        TianyiAdaptTime1 = %s, Reduce Factor = %.2f\n",ts->TianyiAdaptTime1 ? "TRUE":"FALSE",ts->TianyiAdjustFactor1);
  PetscPrintf(PETSC_COMM_WORLD,"                        TianyiAdaptTime2 = %s, Adjust Factor = %.2f\n",ts->TianyiAdaptTime2 ? "TRUE":"FALSE",ts->TianyiAdjustFactor2);
  PetscPrintf(PETSC_COMM_WORLD,"                        min_nits = %d, min_lits = %d\n",ts->min_nits,ts->min_lits);
  PetscPrintf(PETSC_COMM_WORLD,"                        max_nits = %d, max_lits = %d\n",ts->max_nits,ts->max_lits);
  PetscPrintf(PETSC_COMM_WORLD,"                        max_reject = %d\n",ts->max_reject);

  //***************************Pre Simulation Setup***************************//
  // Initialize vectors, when you get time, move the initialize sol_der part into PetIGA and petsd
  Vec U; Vec sol_der;
  ierr = IGACreateVec(iga,&sol_der);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&U);CHKERRQ(ierr);
  ierr = VecZeroEntries(sol_der);CHKERRQ(ierr);
  ts->vec_der = sol_der; // make sure the initial vec_der is not null
  ts->recover = PETSC_TRUE;

  // Update domain size in case read existing
  PetscReal MaxDomain[user.dim],MinDomain[user.dim];
  if (iga->geometry) {
    ierr = IGAComputeCoordExtrema(iga,U,user.dim,&MaxDomain[0],GetGeomtry,1,&user);CHKERRQ(ierr);
    ierr = IGAComputeCoordExtrema(iga,U,user.dim,&MinDomain[0],GetGeomtry,0,&user);CHKERRQ(ierr);
    for (i=0;i<user.dim;i++) {user.DomainSize[i][0] = MinDomain[i]; user.DomainSize[i][1] = MaxDomain[i];}
  }

  // Check if recover initial derivative
  PetscBool recover = PETSC_FALSE;
  sscanf(desiredTime,"%lf",&user.InitialTime);
  user.Condition = file_exist(initial);
  if (file_exist(der_initial)) recover = PETSC_TRUE;

  // Read Initital Condition
  PetscPrintf(PETSC_COMM_WORLD,"----------Read Initial Condition----------\n");
  PetscPrintf(PETSC_COMM_WORLD,"Desired IC File is %s\n",desiredTime);
  if (user.Condition) { /* initial condition from datafile */
    ierr = IGAReadVec(iga,U,initial);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"Solution Read Successfully. Current Time Step %.6e.\n",user.InitialTime);
  } else {                /* initial condition is random */
    PetscPrintf(PETSC_COMM_WORLD,"L2 Projection Forming IC.\n");
    Vec b;
    Mat A;
    KSP ksp;
    ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);

    ierr = IGACreateVec(iga,&b);CHKERRQ(ierr);
    ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);

    //L2 Projection for IC
    ierr = IGASetFormSystem(iga,L2Projection_IC,&user);CHKERRQ(ierr);
    ierr = IGAComputeSystem(iga,A,b);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
    ierr = KSPSolve(ksp,b,U);CHKERRQ(ierr);
    // Clear L2 Projection
    ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = VecDestroy(&b);CHKERRQ(ierr);
  }

  // Read Initial Derivative file
  if (recover){
    ierr = IGAReadVec(iga,sol_der,der_initial);CHKERRQ(ierr);
    ts->vec_der = sol_der; ts->recover = recover;
    PetscPrintf(PETSC_COMM_WORLD,"Time Derivative Successfully. Current Time Step %.6e.\n",user.InitialTime);
  } else {
    PetscPrintf(PETSC_COMM_WORLD,"No initial time derivative file detect. Forming using Backward Euler.\n");
  }

  //***************************Solver***************************//
  PetscPrintf(PETSC_COMM_WORLD,"----------Computation History----------\n");
  ierr = IGAWrite(iga,"GeometryNSK.dat");CHKERRQ(ierr);
  ierr = TSSolve(ts,U);CHKERRQ(ierr);

  //***************************Post Simulation Setup***************************//
  //ierr = IGAWriteVec(iga,U,"2P2C_Solution.dat");CHKERRQ(ierr);
  //ierr = IGADrawVec(iga,U,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  //ierr = VecViewFromOptions(U,NULL,"-view");CHKERRQ(ierr);
  //ierr = VecView(U,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  ierr = VecDestroy(&sol_der);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
