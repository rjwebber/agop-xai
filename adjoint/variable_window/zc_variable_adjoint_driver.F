C=======================================================================
C Runtime-variable segmented reverse of the authentic run23 final3 kernel.
C The number of transitions is read from the independently produced path
C header.  An optional 59,148-value terminal REAL seed computes J^T w;
C without it, the terminal seed is the canonical Nino-3 mean.
C
C Arguments: path file, output prefix, optional terminal-seed stream.
C=======================================================================
      PROGRAM ZC_VARIABLE_ADJOINT_DRIVER
      INCLUDE 'zc_kernel_state.inc'

      REAL STATE_R_INB(ZC_STATE_NREAL)
      REAL STATE_R_OUT(ZC_STATE_NREAL),STATE_R_OUTB(ZC_STATE_NREAL)
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
      INTEGER TO_OFFSET
      PARAMETER (TO_OFFSET=32121)
      REAL NINO3,TSTART,TEND,MAXABS
      DOUBLE PRECISION NORM2,NINO3_SUM,OBJECTIVE_VALUE,XDIFF,XREF
      DOUBLE PRECISION FWD_REAL_DIFF2,FWD_REAL_REF2
      DOUBLE PRECISION FWD_REAL_STEP_MAX,FWD_REAL_STEP_REL
      DOUBLE PRECISION FWD_REAL_WORST_MAX,FWD_REAL_WORST_REL
      DOUBLE PRECISION FWD_COMPLEX_DIFF2,FWD_COMPLEX_REF2
      DOUBLE PRECISION FWD_COMPLEX_STEP_MAX,FWD_COMPLEX_STEP_REL
      DOUBLE PRECISION FWD_COMPLEX_WORST_MAX,FWD_COMPLEX_WORST_REL
      DOUBLE PRECISION REAL_MAX_TOL,REAL_REL_TOL
      DOUBLE PRECISION COMPLEX_MAX_TOL,COMPLEX_REL_TOL
      PARAMETER (REAL_MAX_TOL=1.0D-3,REAL_REL_TOL=1.0D-6)
      PARAMETER (COMPLEX_MAX_TOL=1.0D-4,COMPLEX_REL_TOL=1.0D-6)
      INTEGER I,J,K,N,NSTEPS,IMAX,IOS
      LOGICAL CUSTOM_SEED
      CHARACTER*512 PATH_FILE,OUTPUT_PREFIX,GRADIENT_FILE,META_FILE
      CHARACTER*512 REPLAY_FILE,SEED_FILE

      PATH_FILE=' '
      OUTPUT_PREFIX=' '
      SEED_FILE=' '
      CALL GET_COMMAND_ARGUMENT(1,PATH_FILE)
      CALL GET_COMMAND_ARGUMENT(2,OUTPUT_PREFIX)
      CALL GET_COMMAND_ARGUMENT(3,SEED_FILE)
      IF(LEN_TRIM(PATH_FILE).EQ.0) PATH_FILE='zc_variable_path.bin'
      IF(LEN_TRIM(OUTPUT_PREFIX).EQ.0) OUTPUT_PREFIX='zc_variable'

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

      NINO3_SUM=0.0D0
      DO 35 J=21,31
         DO 34 I=13,18
            K=TO_OFFSET+I+30*(J-1)
            NINO3_SUM=NINO3_SUM+DBLE(STATE_R_PATH(K,NSTEPS))
 34      CONTINUE
 35   CONTINUE
      NINO3=NINO3_SUM/66.0D0
      DO 36 I=1,ZC_STATE_NREAL
         STATE_R_OUTB(I)=0.0
 36   CONTINUE
      CUSTOM_SEED=LEN_TRIM(SEED_FILE).GT.0
      IF(CUSTOM_SEED) THEN
         CALL ZC_VW_READ_REAL(TRIM(SEED_FILE),STATE_R_OUTB,
     A    ZC_STATE_NREAL)
         OBJECTIVE_VALUE=0.0D0
         DO 37 I=1,ZC_STATE_NREAL
            OBJECTIVE_VALUE=OBJECTIVE_VALUE+
     A       DBLE(STATE_R_OUTB(I))*DBLE(STATE_R_PATH(I,NSTEPS))
 37      CONTINUE
      ELSE
         DO 39 J=21,31
            DO 38 I=13,18
               K=TO_OFFSET+I+30*(J-1)
               STATE_R_OUTB(K)=1.0/66.0
 38         CONTINUE
 39      CONTINUE
         OBJECTIVE_VALUE=NINO3
      ENDIF

      CALL CPU_TIME(TSTART)
      FWD_REAL_WORST_MAX=0.0D0
      FWD_REAL_WORST_REL=0.0D0
      FWD_COMPLEX_WORST_MAX=0.0D0
      FWD_COMPLEX_WORST_REL=0.0D0
      REPLAY_FILE=TRIM(OUTPUT_PREFIX)//'_forward_replay.csv'
      OPEN(203,FILE=TRIM(REPLAY_FILE),STATUS='REPLACE',IOSTAT=IOS)
      IF(IOS.NE.0) STOP 'cannot open forward replay output'
      WRITE(203,'(A)') 'step,real_max_abs,real_rel_l2,'//
     A 'complex_max_abs,complex_rel_l2'
      DO 60 N=NSTEPS,1,-1
         CALL ZC_KERNEL_WINDOW_FWD(STATE_R_PATH(1,N-1),
     A    STATE_C_PATH(1,N-1),STATE_D_PATH(1,N-1),
     B    STATE_I_PATH(1,N-1),STATE_TIME_PATH(1,N-1),1,
     C    STATE_R_OUT,STATE_C_OUT,STATE_D_OUT,STATE_I_OUT,
     D    STATE_TIME_OUT,TAPE_WORK)
         FWD_REAL_DIFF2=0.0D0
         FWD_REAL_REF2=0.0D0
         FWD_REAL_STEP_MAX=0.0D0
         DO 41 I=1,ZC_STATE_NREAL
            XDIFF=DBLE(STATE_R_OUT(I))-DBLE(STATE_R_PATH(I,N))
            XREF=DBLE(STATE_R_PATH(I,N))
            IF(XDIFF.NE.XDIFF .OR. XREF.NE.XREF) THEN
               WRITE(*,*) 'nonfinite REAL replay: step/index ',N,I
               STOP 2
            ENDIF
            FWD_REAL_DIFF2=FWD_REAL_DIFF2+XDIFF*XDIFF
            FWD_REAL_REF2=FWD_REAL_REF2+XREF*XREF
            FWD_REAL_STEP_MAX=MAX(FWD_REAL_STEP_MAX,ABS(XDIFF))
 41      CONTINUE
         FWD_REAL_STEP_REL=DSQRT(FWD_REAL_DIFF2/
     A    MAX(FWD_REAL_REF2,1.0D-300))
         FWD_REAL_WORST_MAX=MAX(FWD_REAL_WORST_MAX,FWD_REAL_STEP_MAX)
         FWD_REAL_WORST_REL=MAX(FWD_REAL_WORST_REL,FWD_REAL_STEP_REL)
         FWD_COMPLEX_DIFF2=0.0D0
         FWD_COMPLEX_REF2=0.0D0
         FWD_COMPLEX_STEP_MAX=0.0D0
         DO 42 I=1,ZC_STATE_NCOMPLEX
            XDIFF=DBLE(REAL(STATE_C_OUT(I)))-
     A       DBLE(REAL(STATE_C_PATH(I,N)))
            XREF=DBLE(REAL(STATE_C_PATH(I,N)))
            IF(XDIFF.NE.XDIFF .OR. XREF.NE.XREF) THEN
               WRITE(*,*) 'nonfinite COMPLEX replay: step/index ',N,I
               STOP 2
            ENDIF
            FWD_COMPLEX_DIFF2=FWD_COMPLEX_DIFF2+XDIFF*XDIFF
            FWD_COMPLEX_REF2=FWD_COMPLEX_REF2+XREF*XREF
            FWD_COMPLEX_STEP_MAX=MAX(FWD_COMPLEX_STEP_MAX,ABS(XDIFF))
            XDIFF=DBLE(AIMAG(STATE_C_OUT(I)))-
     A       DBLE(AIMAG(STATE_C_PATH(I,N)))
            XREF=DBLE(AIMAG(STATE_C_PATH(I,N)))
            IF(XDIFF.NE.XDIFF .OR. XREF.NE.XREF) THEN
               WRITE(*,*) 'nonfinite COMPLEX replay: step/index ',N,I
               STOP 2
            ENDIF
            FWD_COMPLEX_DIFF2=FWD_COMPLEX_DIFF2+XDIFF*XDIFF
            FWD_COMPLEX_REF2=FWD_COMPLEX_REF2+XREF*XREF
            FWD_COMPLEX_STEP_MAX=MAX(FWD_COMPLEX_STEP_MAX,ABS(XDIFF))
 42      CONTINUE
         FWD_COMPLEX_STEP_REL=DSQRT(FWD_COMPLEX_DIFF2/
     A    MAX(FWD_COMPLEX_REF2,1.0D-300))
         FWD_COMPLEX_WORST_MAX=MAX(FWD_COMPLEX_WORST_MAX,
     A    FWD_COMPLEX_STEP_MAX)
         FWD_COMPLEX_WORST_REL=MAX(FWD_COMPLEX_WORST_REL,
     A    FWD_COMPLEX_STEP_REL)
         WRITE(203,'(I0,4(",",ES24.16))') N,FWD_REAL_STEP_MAX,
     A    FWD_REAL_STEP_REL,FWD_COMPLEX_STEP_MAX,FWD_COMPLEX_STEP_REL
         DO 46 I=1,ZC_TAPE_WIDTH
            IF(TAPE_WORK(I,1).NE.TAPE_REFERENCE(I,N)) THEN
               WRITE(*,*) 'branch FWD mismatch: step/index ',N,I
               STOP 2
            ENDIF
 46      CONTINUE
         IF(FWD_REAL_STEP_MAX.GT.REAL_MAX_TOL .OR.
     A      FWD_REAL_STEP_REL.GT.REAL_REL_TOL .OR.
     B      FWD_COMPLEX_STEP_MAX.GT.COMPLEX_MAX_TOL .OR.
     C      FWD_COMPLEX_STEP_REL.GT.COMPLEX_REL_TOL) THEN
            WRITE(*,*) 'generated one-step FWD tolerance failure ',N,
     A       FWD_REAL_STEP_MAX,FWD_REAL_STEP_REL,
     B       FWD_COMPLEX_STEP_MAX,FWD_COMPLEX_STEP_REL
            STOP 2
         ENDIF
         CALL ZC_KERNEL_WINDOW_BWD(STATE_R_PATH(1,N-1),STATE_R_INB,
     A    STATE_C_PATH(1,N-1),STATE_D_PATH(1,N-1),
     B    STATE_I_PATH(1,N-1),STATE_TIME_PATH(1,N-1),1,
     C    STATE_R_OUT,STATE_R_OUTB,STATE_C_OUT,STATE_D_PATH(1,N),
     D    STATE_I_PATH(1,N),STATE_TIME_PATH(1,N),TAPE_WORK)
         IF(N.GT.1) THEN
            DO 50 I=1,ZC_STATE_NREAL
               STATE_R_OUTB(I)=STATE_R_INB(I)
 50         CONTINUE
         ENDIF
 60   CONTINUE
      CLOSE(203)
      CALL CPU_TIME(TEND)

      GRADIENT_FILE=TRIM(OUTPUT_PREFIX)//'_gradient.bin'
      CALL ZC_VW_WRITE_REAL(TRIM(GRADIENT_FILE),STATE_R_INB,
     A ZC_STATE_NREAL)
      NORM2=0.0D0
      MAXABS=0.0
      IMAX=0
      DO 20 I=1,ZC_STATE_NREAL
         NORM2=NORM2+DBLE(STATE_R_INB(I))*DBLE(STATE_R_INB(I))
         IF(ABS(STATE_R_INB(I)).GT.MAXABS) THEN
            MAXABS=ABS(STATE_R_INB(I))
            IMAX=I
         ENDIF
 20   CONTINUE
      NORM2=DSQRT(NORM2)
      META_FILE=TRIM(OUTPUT_PREFIX)//'_gradient.txt'
      OPEN(202,FILE=TRIM(META_FILE),STATUS='REPLACE',IOSTAT=IOS)
      IF(IOS.NE.0) STOP 'cannot open gradient metadata output'
      WRITE(202,'(A)') 'method=Tapenade reverse mode'
      WRITE(202,'(A)') 'driver=isolated runtime-variable window'
      IF(CUSTOM_SEED) THEN
         WRITE(202,'(A)')
     A    'objective=custom terminal real-state linear functional'
         WRITE(202,'(A)') 'terminal_seed=custom_float32_stream'
      ELSE
         WRITE(202,'(A)') 'objective=canonical Nino-3 SST anomaly'
         WRITE(202,'(A)') 'terminal_seed=canonical_nino3'
      ENDIF
      WRITE(202,'(A)') 'fortran_box=I=13:18,J=21:31'
      WRITE(202,'(A,I0)') 'transitions=',NSTEPS
      WRITE(202,'(A)') 'state_semantics=pre-input checkpoint'
      WRITE(202,'(A)') 'reverse_schedule=one-step FWD/BWD pairs'
      WRITE(202,'(A)') 'reference_path=independent authentic primal'
      WRITE(202,'(A,I0)') 'real_state_length=',ZC_STATE_NREAL
      WRITE(202,'(A,ES24.16)') 'final_nino3_celsius=',NINO3
      WRITE(202,'(A,ES24.16)') 'objective_value=',OBJECTIVE_VALUE
      WRITE(202,'(A,ES24.16)') 'cpu_seconds=',TEND-TSTART
      WRITE(202,'(A,ES24.16)') 'fwd_real_worst_max_abs=',
     A FWD_REAL_WORST_MAX
      WRITE(202,'(A,ES24.16)') 'fwd_real_worst_rel_l2=',
     A FWD_REAL_WORST_REL
      WRITE(202,'(A,ES24.16)') 'fwd_complex_worst_max_abs=',
     A FWD_COMPLEX_WORST_MAX
      WRITE(202,'(A,ES24.16)') 'fwd_complex_worst_rel_l2=',
     A FWD_COMPLEX_WORST_REL
      WRITE(202,'(A,ES24.16)') 'gradient_l2_norm=',NORM2
      WRITE(202,'(A,ES24.16)') 'gradient_max_abs=',MAXABS
      WRITE(202,'(A,I0)') 'gradient_max_abs_fortran_index=',IMAX
      CLOSE(202)
      WRITE(*,'(A,I0)') 'Transitions: ',NSTEPS
      WRITE(*,'(A,ES16.8)') 'Final canonical Nino-3 (C): ',NINO3
      WRITE(*,'(A,ES16.8)') 'Terminal objective value:   ',
     A OBJECTIVE_VALUE
      WRITE(*,'(A,ES16.8)') 'Reverse CPU time (s):       ',TEND-TSTART
      WRITE(*,'(A,A)') 'Gradient: ',TRIM(GRADIENT_FILE)
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
