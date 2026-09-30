C=======================================================================
C Runtime-variable tangent checker paired with zc_variable_adjoint_driver.
C It re-anchors the unchanged generated one-step tangent at every boundary
C from the authentic path, matching the reverse schedule exactly.
C
C Arguments: path file, initial REAL direction, output prefix.
C=======================================================================
      PROGRAM ZC_VARIABLE_TANGENT_DRIVER
      INCLUDE 'zc_kernel_state.inc'

      REAL STATE_R_IN_D(ZC_STATE_NREAL),STATE_R_OUT_D(ZC_STATE_NREAL)
      REAL STATE_R_OUT(ZC_STATE_NREAL)
      COMPLEX STATE_C_OUT(ZC_STATE_NCOMPLEX)
      REAL*8 STATE_D_OUT(ZC_STATE_NDOUBLE)
      INTEGER STATE_I_OUT(ZC_STATE_NINTEGER)
      REAL STATE_TIME_OUT(ZC_STATE_NTIME)
      REAL, ALLOCATABLE :: STATE_R_PATH(:,:)
      COMPLEX, ALLOCATABLE :: STATE_C_PATH(:,:)
      REAL*8, ALLOCATABLE :: STATE_D_PATH(:,:)
      INTEGER, ALLOCATABLE :: STATE_I_PATH(:,:)
      REAL, ALLOCATABLE :: STATE_TIME_PATH(:,:)
      INTEGER, ALLOCATABLE :: TAPE_REFERENCE(:,:)
      INTEGER TAPE_WORK(ZC_TAPE_WIDTH,1)
      DOUBLE PRECISION NORM2,XDIFF,XREF,DIFF2,REF2
      DOUBLE PRECISION STEP_MAX,STEP_REL,WORST_MAX,WORST_REL
      DOUBLE PRECISION REAL_MAX_TOL,REAL_REL_TOL
      PARAMETER (REAL_MAX_TOL=1.0D-3,REAL_REL_TOL=1.0D-6)
      INTEGER I,N,NSTEPS,IOS
      REAL TSTART,TEND
      CHARACTER*512 PATH_FILE,DIRECTION_FILE,OUTPUT_PREFIX
      CHARACTER*512 OUTPUT_FILE,META_FILE

      PATH_FILE=' '
      DIRECTION_FILE=' '
      OUTPUT_PREFIX=' '
      CALL GET_COMMAND_ARGUMENT(1,PATH_FILE)
      CALL GET_COMMAND_ARGUMENT(2,DIRECTION_FILE)
      CALL GET_COMMAND_ARGUMENT(3,OUTPUT_PREFIX)
      IF(LEN_TRIM(PATH_FILE).EQ.0) PATH_FILE='zc_variable_path.bin'
      IF(LEN_TRIM(DIRECTION_FILE).EQ.0)
     A DIRECTION_FILE='tangent_direction.bin'
      IF(LEN_TRIM(OUTPUT_PREFIX).EQ.0) OUTPUT_PREFIX='zc_variable_tangent'

      CALL ZC_VW_INITIALIZE
      CALL ZC_VW_READ_PATH_HEADER(TRIM(PATH_FILE),NSTEPS)
      ALLOCATE(STATE_R_PATH(ZC_STATE_NREAL,0:NSTEPS),STAT=IOS)
      IF(IOS.NE.0) STOP 'cannot allocate real path'
      ALLOCATE(STATE_C_PATH(ZC_STATE_NCOMPLEX,0:NSTEPS),STAT=IOS)
      IF(IOS.NE.0) STOP 'cannot allocate complex path'
      ALLOCATE(STATE_D_PATH(ZC_STATE_NDOUBLE,0:NSTEPS),STAT=IOS)
      IF(IOS.NE.0) STOP 'cannot allocate double path'
      ALLOCATE(STATE_I_PATH(ZC_STATE_NINTEGER,0:NSTEPS),STAT=IOS)
      IF(IOS.NE.0) STOP 'cannot allocate integer path'
      ALLOCATE(STATE_TIME_PATH(ZC_STATE_NTIME,0:NSTEPS),STAT=IOS)
      IF(IOS.NE.0) STOP 'cannot allocate time path'
      ALLOCATE(TAPE_REFERENCE(ZC_TAPE_WIDTH,NSTEPS),STAT=IOS)
      IF(IOS.NE.0) STOP 'cannot allocate branch path'
      CALL ZC_VW_READ_PATH(TRIM(PATH_FILE),NSTEPS,STATE_R_PATH,
     A STATE_C_PATH,STATE_D_PATH,STATE_I_PATH,STATE_TIME_PATH,
     B TAPE_REFERENCE)
      CALL ZC_VW_READ_REAL(TRIM(DIRECTION_FILE),STATE_R_IN_D,
     A ZC_STATE_NREAL)

      WORST_MAX=0.0D0
      WORST_REL=0.0D0
      CALL CPU_TIME(TSTART)
      DO 100 N=1,NSTEPS
         CALL ZC_KERNEL_WINDOW_D(STATE_R_PATH(1,N-1),STATE_R_IN_D,
     A    STATE_C_PATH(1,N-1),STATE_D_PATH(1,N-1),
     B    STATE_I_PATH(1,N-1),STATE_TIME_PATH(1,N-1),1,
     C    STATE_R_OUT,STATE_R_OUT_D,STATE_C_OUT,STATE_D_OUT,
     D    STATE_I_OUT,STATE_TIME_OUT,TAPE_WORK)
         DIFF2=0.0D0
         REF2=0.0D0
         STEP_MAX=0.0D0
         DO 70 I=1,ZC_STATE_NREAL
            XDIFF=DBLE(STATE_R_OUT(I))-DBLE(STATE_R_PATH(I,N))
            XREF=DBLE(STATE_R_PATH(I,N))
            IF(XDIFF.NE.XDIFF .OR. XREF.NE.XREF) THEN
               WRITE(*,*) 'nonfinite tangent replay: step/index ',N,I
               STOP 2
            ENDIF
            DIFF2=DIFF2+XDIFF*XDIFF
            REF2=REF2+XREF*XREF
            STEP_MAX=MAX(STEP_MAX,ABS(XDIFF))
 70      CONTINUE
         STEP_REL=DSQRT(DIFF2/MAX(REF2,1.0D-300))
         WORST_MAX=MAX(WORST_MAX,STEP_MAX)
         WORST_REL=MAX(WORST_REL,STEP_REL)
         IF(STEP_MAX.GT.REAL_MAX_TOL .OR. STEP_REL.GT.REAL_REL_TOL)
     A   THEN
            WRITE(*,*) 'generated tangent FWD tolerance failure ',N,
     A       STEP_MAX,STEP_REL
            STOP 2
         ENDIF
         DO 80 I=1,ZC_TAPE_WIDTH
            IF(TAPE_WORK(I,1).NE.TAPE_REFERENCE(I,N)) THEN
               WRITE(*,*) 'tangent branch mismatch: step/index ',N,I
               STOP 2
            ENDIF
 80      CONTINUE
         DO 90 I=1,ZC_STATE_NREAL
            STATE_R_IN_D(I)=STATE_R_OUT_D(I)
 90      CONTINUE
 100  CONTINUE
      CALL CPU_TIME(TEND)

      NORM2=0.0D0
      DO 130 I=1,ZC_STATE_NREAL
         NORM2=NORM2+DBLE(STATE_R_IN_D(I))*DBLE(STATE_R_IN_D(I))
 130  CONTINUE
      NORM2=DSQRT(NORM2)
      OUTPUT_FILE=TRIM(OUTPUT_PREFIX)//'_jv.bin'
      CALL ZC_VW_WRITE_REAL(TRIM(OUTPUT_FILE),STATE_R_IN_D,
     A ZC_STATE_NREAL)
      META_FILE=TRIM(OUTPUT_PREFIX)//'_jv.txt'
      OPEN(202,FILE=TRIM(META_FILE),STATUS='REPLACE',IOSTAT=IOS)
      IF(IOS.NE.0) STOP 'cannot open tangent metadata output'
      WRITE(202,'(A)') 'method=Tapenade tangent mode'
      WRITE(202,'(A)') 'driver=isolated runtime-variable window'
      WRITE(202,'(A)') 'schedule=authentic one-step boundary composition'
      WRITE(202,'(A,I0)') 'transitions=',NSTEPS
      WRITE(202,'(A,ES24.16)') 'jv_l2_norm=',NORM2
      WRITE(202,'(A,ES24.16)') 'cpu_seconds=',TEND-TSTART
      WRITE(202,'(A,ES24.16)') 'fwd_real_worst_max_abs=',WORST_MAX
      WRITE(202,'(A,ES24.16)') 'fwd_real_worst_rel_l2=',WORST_REL
      CLOSE(202)
      WRITE(*,'(A,I0)') 'Transitions: ',NSTEPS
      WRITE(*,'(A,A)') 'Tangent: ',TRIM(OUTPUT_FILE)
      STOP
      END


C     Passive complex-workspace helpers expected by the generated wrapper.
      SUBROUTINE ZC_PACK_COMPLEX(A,N,X,K)
      COMPLEX A(N),X(*)
      DO 10 I=1,N
         X(K)=A(I)
         K=K+1
 10   CONTINUE
      RETURN
      END

      SUBROUTINE ZC_UNPACK_COMPLEX(A,N,X,K)
      COMPLEX A(N),X(*)
      DO 10 I=1,N
         A(I)=X(K)
         K=K+1
 10   CONTINUE
      RETURN
      END


C     GETN is a discrete branch selector and remains passive.
      SUBROUTINE GETN(NSST)
      INCLUDE 'zeq.common'
      COMMON/ZDATA/WM(30,34,2,12),DIVM(30,34,12),Q0O(30,34),
     A SSTM(30,34,12),UO(30,34),VO(30,34),DO(30,34),HTAU(34,30,2),
     B WEM(30,34,12),TO(30,34),UV(30,34,2,12),US(30,34),VS(30,34),
     C WP(30,34),DT1(30,34,7),TT(30,34),UV1(30,34),UV2(30,34),
     D WM1(30,34),UAT(30,34),VAT(30,34),DIVT(30,34)
      COMMON/SGRID/XWRS,DXRS,YNRS,DYRS,NYPS
      SAVE
      DX1=DXRS*1.11E7
      DY1=DYRS*1.11E7
      DTT=DTD*2.628E6
      NSST=2
      DO 20 I=6,25
         DO 10 J=6,32
            AAA=ABS(US(I,J))*DTT/DX1
            BBB=ABS(VS(I,J))*DTT/DY1
            IAA=1+AAA
            IBB=1+BBB
            NSST=MAX0(NSST,IAA)
            NSST=MAX0(NSST,IBB)
 10      CONTINUE
 20   CONTINUE
      RETURN
      END
