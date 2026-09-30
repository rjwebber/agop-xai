      PROGRAM TEST_MLOOP_AD
      IMPLICIT NONE
      INCLUDE 'test_common.inc'

      INTEGER I,J,K
      REAL DAKB(85),DBK(85),DHB(117,85),DUB(117,85)
      REAL DF(117,85),DG(117,85),DUBNDY(117,4),DHBNDY(117,4)
      REAL WAK(85),WAKB(85),WHB(117,85),WUB(117,85)
      REAL WU(117,85),WH(117,85),WV(117,85)
      REAL WUBNDY(117,4),WHBNDY(117,4)
      REAL EPS
      REAL*8 LHS,RHS,SCALE,RELDEF,JP,JM,FDIRECTION,FDERR

      CALL INITIALIZE_TEST_VECTORS

C     Tangent product J*v.
      CALL RESET_PRIMAL
      CALL LOAD_TANGENT
      CALL MLOOP_D
      LHS = OUTPUT_TANGENT_DOT()

C     Reverse product J^T*w.  MLOOP_B performs its required primal trace.
      CALL RESET_PRIMAL
      CALL LOAD_ADJOINT_SEED
      CALL MLOOP_B
      RHS = INPUT_DIRECTION_DOT()
      SCALE = MAX(1.0D0,ABS(LHS),ABS(RHS))
      RELDEF = ABS(LHS-RHS)/SCALE

C     Independent centered finite difference of the untouched primal MLOOP.
      EPS = 1.0E-2
      CALL RESET_PRIMAL
      CALL PERTURB_INPUTS(EPS)
      CALL MLOOP
      JP = OUTPUT_OBJECTIVE()
      CALL RESET_PRIMAL
      CALL PERTURB_INPUTS(-EPS)
      CALL MLOOP
      JM = OUTPUT_OBJECTIVE()
      FDIRECTION = (JP-JM)/(2.0D0*DBLE(EPS))
      FDERR = ABS(FDIRECTION-LHS)/MAX(1.0D0,ABS(FDIRECTION),
     A ABS(LHS))

      WRITE(*,'(A,ES24.16)') 'tangent_dot=',LHS
      WRITE(*,'(A,ES24.16)') 'adjoint_dot=',RHS
      WRITE(*,'(A,ES24.16)') 'dot_relative_defect=',RELDEF
      WRITE(*,'(A,ES24.16)') 'centered_fd=',FDIRECTION
      WRITE(*,'(A,ES24.16)') 'fd_relative_error=',FDERR

      IF (RELDEF.GT.2.0D-5) STOP 2
      IF (FDERR.GT.2.0D-5) STOP 3
      WRITE(*,'(A)') 'PASS mloop tangent/adjoint/finite-difference test'

      CONTAINS

      SUBROUTINE INITIALIZE_TEST_VECTORS
      DO I=1,85
        DAKB(I)=SIN(0.13*FLOAT(I))
        DBK(I)=COS(0.17*FLOAT(I))
        WAK(I)=SIN(0.19*FLOAT(I))
        WAKB(I)=COS(0.23*FLOAT(I))
        DO J=1,117
          DHB(J,I)=SIN(0.013*FLOAT(3*J+I))
          DUB(J,I)=COS(0.011*FLOAT(J+2*I))
          DF(J,I)=SIN(0.017*FLOAT(2*J+3*I))
          DG(J,I)=COS(0.019*FLOAT(4*J+I))
          WHB(J,I)=SIN(0.007*FLOAT(J+5*I))
          WUB(J,I)=COS(0.009*FLOAT(2*J+I))
          WU(J,I)=SIN(0.014*FLOAT(J+I))
          WH(J,I)=COS(0.012*FLOAT(J+3*I))
          WV(J,I)=SIN(0.016*FLOAT(3*J+2*I))
        END DO
      END DO
      DO K=1,4
        DO J=1,117
          DUBNDY(J,K)=SIN(0.021*FLOAT(J+K))
          DHBNDY(J,K)=COS(0.022*FLOAT(J+2*K))
          WUBNDY(J,K)=SIN(0.025*FLOAT(2*J+K))
          WHBNDY(J,K)=COS(0.027*FLOAT(J+3*K))
        END DO
      END DO
      END SUBROUTINE INITIALIZE_TEST_VECTORS

      SUBROUTINE RESET_PRIMAL
      NX=78
      NY=115
      HEQUIV=86.0
      TZERO=0.5
      TENDD=10.5
      DTD=1.0/3.0
      XWD=124.0
      XED=280.0
      YSD=-28.75
      YND=28.75
      TDECAY=30.0
      TPLMIN=-1000.0
      NPRINT=999999
      NSEG=1
      YNORTH(1)=28.75
      YSOUTH(1)=-28.75
      XWEST(1)=124.0
      NTAPE=0
      NREWND=11
      NATM=2
      NATMR=2
      NT=1
      TD=0.5
      ISTART=1
      PRINT_TIMESTEP_DIAGNOSTICS=.FALSE.
      DONT_DO_W_BNDRY_MASS_CLOSURE=.FALSE.
      CALL CONSTC
      ISTART=1
      DO I=1,85
        AK(I)=1.0E-3*SIN(0.031*FLOAT(I))
        AKB(I)=1.0E-3*COS(0.029*FLOAT(I))
        BK(I)=1.0E-4*SIN(0.037*FLOAT(I))
        DO J=1,117
          HB(J,I)=1.0E-3*SIN(0.003*FLOAT(J+I))
          UB(J,I)=1.0E-3*COS(0.004*FLOAT(2*J+I))
          U(J,I)=1.0E-3*SIN(0.005*FLOAT(J+3*I))
          H(J,I)=1.0E-3*COS(0.006*FLOAT(2*J+I))
          V(J,I)=1.0E-3*SIN(0.007*FLOAT(J+I))
          F(J,I)=1.0E-4*COS(0.008*FLOAT(J+2*I))
          G(J,I)=1.0E-4*SIN(0.009*FLOAT(3*J+I))
        END DO
      END DO
      DO K=1,4
        DO J=1,117
          UBNDY(J,K)=1.0E-3*SIN(0.01*FLOAT(J+K))
          HBNDY(J,K)=1.0E-3*COS(0.01*FLOAT(J+2*K))
        END DO
      END DO
      END SUBROUTINE RESET_PRIMAL

      SUBROUTINE ZERO_TANGENT
      AKD=0.0
      AKBD=0.0
      BKD=0.0
      HBD=0.0
      UBD=0.0
      UD=0.0
      HD=0.0
      VD=0.0
      FD=0.0
      GD=0.0
      DD=0.0
      R1D=0.0
      R3D=0.0
      GINTD=0.0
      TRD=0.0
      ROSSD=0.0
      UBNDYD=0.0
      HBNDYD=0.0
      END SUBROUTINE ZERO_TANGENT

      SUBROUTINE LOAD_TANGENT
      CALL ZERO_TANGENT
      AKBD=DAKB
      BKD=DBK
      HBD=DHB
      UBD=DUB
      FD=DF
      GD=DG
      UBNDYD=DUBNDY
      HBNDYD=DHBNDY
      END SUBROUTINE LOAD_TANGENT

      SUBROUTINE ZERO_ADJOINT
      AKB0=0.0
      AKBB=0.0
      BKB=0.0
      HBB=0.0
      UBB=0.0
      UB0=0.0
      HB0=0.0
      VB=0.0
      FB=0.0
      GB=0.0
      DB=0.0
      R1B=0.0
      R3B=0.0
      GINTB=0.0
      TRB=0.0
      ROSSB=0.0
      UBNDYB=0.0
      HBNDYB=0.0
      END SUBROUTINE ZERO_ADJOINT

      SUBROUTINE LOAD_ADJOINT_SEED
      CALL ZERO_ADJOINT
      AKB0=WAK
      AKBB=WAKB
      HBB=WHB
      UBB=WUB
      UB0=WU
      HB0=WH
      VB=WV
      UBNDYB=WUBNDY
      HBNDYB=WHBNDY
      END SUBROUTINE LOAD_ADJOINT_SEED

      SUBROUTINE PERTURB_INPUTS(S)
      REAL S
      AKB=AKB+S*DAKB
      BK=BK+S*DBK
      HB=HB+S*DHB
      UB=UB+S*DUB
      F=F+S*DF
      G=G+S*DG
      UBNDY=UBNDY+S*DUBNDY
      HBNDY=HBNDY+S*DHBNDY
      END SUBROUTINE PERTURB_INPUTS

      REAL*8 FUNCTION OUTPUT_TANGENT_DOT()
      OUTPUT_TANGENT_DOT=0.0D0
      DO I=1,85
        OUTPUT_TANGENT_DOT=OUTPUT_TANGENT_DOT+
     A    DBLE(WAK(I))*DBLE(AKD(I))+
     B    DBLE(WAKB(I))*DBLE(AKBD(I))
        DO J=1,117
          OUTPUT_TANGENT_DOT=OUTPUT_TANGENT_DOT+
     A      DBLE(WHB(J,I))*DBLE(HBD(J,I))+
     B      DBLE(WUB(J,I))*DBLE(UBD(J,I))+
     C      DBLE(WU(J,I))*DBLE(UD(J,I))+
     D      DBLE(WH(J,I))*DBLE(HD(J,I))+
     E      DBLE(WV(J,I))*DBLE(VD(J,I))
        END DO
      END DO
      DO K=1,4
        DO J=1,117
          OUTPUT_TANGENT_DOT=OUTPUT_TANGENT_DOT+
     A      DBLE(WUBNDY(J,K))*DBLE(UBNDYD(J,K))+
     B      DBLE(WHBNDY(J,K))*DBLE(HBNDYD(J,K))
        END DO
      END DO
      END FUNCTION OUTPUT_TANGENT_DOT

      REAL*8 FUNCTION INPUT_DIRECTION_DOT()
      INPUT_DIRECTION_DOT=0.0D0
      DO I=1,85
        INPUT_DIRECTION_DOT=INPUT_DIRECTION_DOT+
     A    DBLE(DAKB(I))*DBLE(AKBB(I))+
     B    DBLE(DBK(I))*DBLE(BKB(I))
        DO J=1,117
          INPUT_DIRECTION_DOT=INPUT_DIRECTION_DOT+
     A      DBLE(DHB(J,I))*DBLE(HBB(J,I))+
     B      DBLE(DUB(J,I))*DBLE(UBB(J,I))+
     C      DBLE(DF(J,I))*DBLE(FB(J,I))+
     D      DBLE(DG(J,I))*DBLE(GB(J,I))
        END DO
      END DO
      DO K=1,4
        DO J=1,117
          INPUT_DIRECTION_DOT=INPUT_DIRECTION_DOT+
     A      DBLE(DUBNDY(J,K))*DBLE(UBNDYB(J,K))+
     B      DBLE(DHBNDY(J,K))*DBLE(HBNDYB(J,K))
        END DO
      END DO
      END FUNCTION INPUT_DIRECTION_DOT

      REAL*8 FUNCTION OUTPUT_OBJECTIVE()
      OUTPUT_OBJECTIVE=0.0D0
      DO I=1,85
        OUTPUT_OBJECTIVE=OUTPUT_OBJECTIVE+
     A    DBLE(WAK(I))*DBLE(AK(I))+
     B    DBLE(WAKB(I))*DBLE(AKB(I))
        DO J=1,117
          OUTPUT_OBJECTIVE=OUTPUT_OBJECTIVE+
     A      DBLE(WHB(J,I))*DBLE(HB(J,I))+
     B      DBLE(WUB(J,I))*DBLE(UB(J,I))+
     C      DBLE(WU(J,I))*DBLE(U(J,I))+
     D      DBLE(WH(J,I))*DBLE(H(J,I))+
     E      DBLE(WV(J,I))*DBLE(V(J,I))
        END DO
      END DO
      DO K=1,4
        DO J=1,117
          OUTPUT_OBJECTIVE=OUTPUT_OBJECTIVE+
     A      DBLE(WUBNDY(J,K))*DBLE(UBNDY(J,K))+
     B      DBLE(WHBNDY(J,K))*DBLE(HBNDY(J,K))
        END DO
      END DO
      END FUNCTION OUTPUT_OBJECTIVE

      END PROGRAM TEST_MLOOP_AD

