C=======================================================================
C User-facing driver for the Tapenade reverse of the authentic ZC
C 31-transition, canonical Nino-3 forecast.  Run it in a certified kernel
C replay directory so fc.data, the namelists, Data/, and zeq9fsu.hst are
C available.  Argument 1 is the certified primal path file; argument 2 is
C an optional output prefix; argument 3 is an optional raw stream of 59,148
C REAL terminal weights.  With no third argument, the terminal weights are
C the canonical Nino-3 average.  With one, the result is J^T w for that
C arbitrary terminal real-state cotangent.
C=======================================================================
      PROGRAM ZC_NINO3_ADJOINT_DRIVER
      INCLUDE 'zc_kernel_state.inc'
      INCLUDE 'zeq.common'
      INCLUDE 'modified_means.common'

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
      INTEGER TARGET_STEPS,TO_OFFSET
      PARAMETER (TARGET_STEPS=31,TO_OFFSET=32121)
      INTEGER TAPE_REFERENCE(ZC_TAPE_WIDTH,TARGET_STEPS)
      INTEGER TAPE_WORK(ZC_TAPE_WIDTH,1)
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
      INTEGER I,N,IMAX,NSTART,NT_RESTART,NATMR,IT1,IOS
      REAL TFIND,TD_RESTART,TZERO_RESTART
      LOGICAL ZC_KERNEL_MODE,CUSTOM_SEED
      INTEGER*8 SEED_BYTES,EXPECTED_SEED_BYTES
      COMMON/commTIME/IT,MP,TY
      COMMON/IST/ist
      COMMON/STRPAR/RSTR,TCUT,TSEED,TAMP1,TAMP2,JLEFT,JRT,ITOP,
     A IBOT,NIC
      COMMON/zc_kernel_mode/ZC_KERNEL_MODE
      CHARACTER*1 HEADER_LOCAL(80)
      CHARACTER*512 PATH_FILE,OUTPUT_PREFIX,GRADIENT_FILE,META_FILE
      CHARACTER*512 REPLAY_FILE,SEED_FILE

      PATH_FILE=' '
      OUTPUT_PREFIX=' '
      SEED_FILE=' '
      CALL GET_COMMAND_ARGUMENT(1,PATH_FILE)
      CALL GET_COMMAND_ARGUMENT(2,OUTPUT_PREFIX)
      CALL GET_COMMAND_ARGUMENT(3,SEED_FILE)
      IF(LEN_TRIM(PATH_FILE).EQ.0) THEN
         PATH_FILE='zc_31step_path.bin'
      ENDIF
      IF(LEN_TRIM(OUTPUT_PREFIX).EQ.0) OUTPUT_PREFIX='zc_nino3'

C     Recreate the certified static model configuration.  The explicit
C     typed state is loaded only after these immutable arrays are built.
      ZC_KERNEL_MODE=.TRUE.
      iprint_namelists=0
      ist=0
      OPEN(5,FILE='fc.data',STATUS='OLD',IOSTAT=IOS)
      IF(IOS.NE.0) THEN
         WRITE(*,*) 'ERROR: cannot open fc.data; IOSTAT=',IOS
         STOP 2
      ENDIF
      READ(5,450) HEADER_LOCAL
 450  FORMAT(80A1)
      CALL OPENFL
      NATM=2
      READ(5,40,END=999) NSTART,TFIND
 40   FORMAT(8X,I15/8X,F15.0)
      IF(NSTART.NE.3) THEN
         WRITE(*,*) 'ERROR: adjoint driver requires NSTART=3'
         STOP 2
      ENDIF
      CALL RDHIST(0,TFIND)
      TD_RESTART=TD
      NT_RESTART=NT
      TZERO_RESTART=TZERO
      NATMR=NATM
      CALL SETUP
      TD=TD_RESTART
      NT=NT_RESTART
      TZERO=TZERO_RESTART
      IT1=TD+120.501
      IT=MOD(IT1,12)
      MP=IT+1
      IF(IT.EQ.0) IT=12
      TY=TD+120.501-IT1
      CALL SETUP2
      CALL INITDAT
      CALL CONSTC
      IF(NIC.NE.0) THEN
         WRITE(*,*) 'ERROR: adjoint driver requires deterministic NIC=0'
         STOP 2
      ENDIF

C     Read the independently generated authentic-primal boundary path.  The
C     reverse below pairs each one-step generated FWD immediately with BWD,
C     bounding tape memory and avoiding stateful checkpoint recomputation.
      ALLOCATE(STATE_R_PATH(ZC_STATE_NREAL,0:TARGET_STEPS))
      ALLOCATE(STATE_C_PATH(ZC_STATE_NCOMPLEX,0:TARGET_STEPS))
      ALLOCATE(STATE_D_PATH(ZC_STATE_NDOUBLE,0:TARGET_STEPS))
      ALLOCATE(STATE_I_PATH(ZC_STATE_NINTEGER,0:TARGET_STEPS))
      ALLOCATE(STATE_TIME_PATH(ZC_STATE_NTIME,0:TARGET_STEPS))
      CALL ZC_AD_READ_PATH(TRIM(PATH_FILE),STATE_R_PATH,STATE_C_PATH,
     A STATE_D_PATH,STATE_I_PATH,STATE_TIME_PATH,TAPE_REFERENCE)

      NINO3_SUM=0.0D0
      DO 35 J=21,31
         DO 34 I=13,18
            K=TO_OFFSET+I+30*(J-1)
            NINO3_SUM=NINO3_SUM+DBLE(STATE_R_PATH(K,TARGET_STEPS))
 34      CONTINUE
 35   CONTINUE
      NINO3=NINO3_SUM/66.0D0
      DO 36 I=1,ZC_STATE_NREAL
         STATE_R_OUTB(I)=0.0
 36   CONTINUE
      CUSTOM_SEED=LEN_TRIM(SEED_FILE).GT.0
      IF(CUSTOM_SEED) THEN
         EXPECTED_SEED_BYTES=4_8*ZC_STATE_NREAL
         INQUIRE(FILE=TRIM(SEED_FILE),SIZE=SEED_BYTES)
         IF(SEED_BYTES.NE.EXPECTED_SEED_BYTES) THEN
            WRITE(*,*) 'ERROR: terminal seed bytes=',SEED_BYTES,
     A       ' expected=',EXPECTED_SEED_BYTES
            STOP 2
         ENDIF
         OPEN(204,FILE=TRIM(SEED_FILE),FORM='UNFORMATTED',
     A    ACCESS='STREAM',STATUS='OLD')
         READ(204) STATE_R_OUTB
         CLOSE(204)
         OBJECTIVE_VALUE=0.0D0
         DO 37 I=1,ZC_STATE_NREAL
            IF(STATE_R_OUTB(I).NE.STATE_R_OUTB(I) .OR.
     A         ABS(STATE_R_OUTB(I)).GT.HUGE(STATE_R_OUTB(I)))
     B      THEN
               WRITE(*,*) 'ERROR: terminal seed contains nonfinite values'
               STOP 2
            ENDIF
            OBJECTIVE_VALUE=OBJECTIVE_VALUE+
     A       DBLE(STATE_R_OUTB(I))*DBLE(STATE_R_PATH(I,TARGET_STEPS))
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
      OPEN(203,FILE=TRIM(REPLAY_FILE),STATUS='REPLACE')
      WRITE(203,'(A)') 'step,real_max_abs,real_rel_l2,'//
     A 'complex_max_abs,complex_rel_l2'
      DO 60 N=TARGET_STEPS,1,-1
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
C        Tapenade correctly prunes passive output copies (DSEED, clock,
C        season indices); their exact canonical values remain in the path and
C        are supplied as passive inputs at every re-anchored step.
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
      OPEN(201,FILE=TRIM(GRADIENT_FILE),FORM='UNFORMATTED',
     A ACCESS='STREAM',STATUS='REPLACE')
      WRITE(201) STATE_R_INB
      CLOSE(201)

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
      OPEN(202,FILE=TRIM(META_FILE),STATUS='REPLACE')
      WRITE(202,'(A)') 'method=Tapenade reverse mode'
      IF(CUSTOM_SEED) THEN
         WRITE(202,'(A)')
     A    'objective=custom terminal real-state linear functional'
         WRITE(202,'(A)') 'terminal_seed=custom_float32_stream'
      ELSE
         WRITE(202,'(A)') 'objective=canonical Nino-3 SST anomaly'
         WRITE(202,'(A)') 'terminal_seed=canonical_nino3'
      ENDIF
      WRITE(202,'(A)') 'fortran_box=I=13:18,J=21:31'
      WRITE(202,'(A)') 'transitions=31'
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
      WRITE(*,'(A,ES16.8)') 'Final canonical Nino-3 (C): ',NINO3
      WRITE(*,'(A,ES16.8)') 'Terminal objective value:   ',
     A OBJECTIVE_VALUE
      WRITE(*,'(A,ES16.8)') 'Reverse CPU time (s):       ',TEND-TSTART
      WRITE(*,'(A,A)') 'Gradient: ',TRIM(GRADIENT_FILE)
      WRITE(*,'(A,A)') 'Metadata: ',TRIM(META_FILE)
      STOP
 999  WRITE(*,*) 'ERROR: unexpected end of fc.data'
      STOP 2
      END


      SUBROUTINE ZC_AD_READ_PATH(NAME,STATE_R,STATE_C,STATE_D,
     A STATE_I,STATE_TIME,TAPE)
      INCLUDE 'zc_kernel_state.inc'
      INTEGER TARGET_STEPS
      PARAMETER (TARGET_STEPS=31)
      CHARACTER*(*) NAME
      REAL STATE_R(ZC_STATE_NREAL,0:TARGET_STEPS)
      COMPLEX STATE_C(ZC_STATE_NCOMPLEX,0:TARGET_STEPS)
      REAL*8 STATE_D(ZC_STATE_NDOUBLE,0:TARGET_STEPS)
      INTEGER STATE_I(ZC_STATE_NINTEGER,0:TARGET_STEPS)
      REAL STATE_TIME(ZC_STATE_NTIME,0:TARGET_STEPS)
      INTEGER TAPE(ZC_TAPE_WIDTH,TARGET_STEPS)
      INTEGER HSTEPS,HREAL,HCOMPLEX,HDOUBLE,HINTEGER,HTIME,HTAPE,IOS
      INTEGER*8 FILE_BYTES,EXPECTED_BYTES
      LOGICAL EXISTS
      EXPECTED_BYTES=7_8*4_8+
     A (TARGET_STEPS+1_8)*(4_8*ZC_STATE_NREAL+
     B  8_8*ZC_STATE_NCOMPLEX+8_8*ZC_STATE_NDOUBLE+
     C  4_8*ZC_STATE_NINTEGER+4_8*ZC_STATE_NTIME)+
     D TARGET_STEPS*4_8*ZC_TAPE_WIDTH
      INQUIRE(FILE=NAME,EXIST=EXISTS,SIZE=FILE_BYTES)
      IF(.NOT.EXISTS .OR. FILE_BYTES.NE.EXPECTED_BYTES) THEN
         WRITE(*,*) 'ERROR: invalid path-file bytes: ',FILE_BYTES,
     A    ' expected ',EXPECTED_BYTES
         STOP 2
      ENDIF
      OPEN(200,FILE=NAME,FORM='UNFORMATTED',ACCESS='STREAM',
     A STATUS='OLD',IOSTAT=IOS)
      IF(IOS.NE.0) THEN
         WRITE(*,*) 'ERROR: cannot open path file; IOSTAT=',IOS
         STOP 2
      ENDIF
      READ(200,IOSTAT=IOS) HSTEPS,HREAL,HCOMPLEX,HDOUBLE,HINTEGER,
     A HTIME,HTAPE
      IF(IOS.NE.0) THEN
         WRITE(*,*) 'ERROR: cannot read path header; IOSTAT=',IOS
         STOP 2
      ENDIF
      IF(HSTEPS.NE.TARGET_STEPS .OR. HREAL.NE.ZC_STATE_NREAL .OR.
     A HCOMPLEX.NE.ZC_STATE_NCOMPLEX .OR.
     B HDOUBLE.NE.ZC_STATE_NDOUBLE .OR.
     C HINTEGER.NE.ZC_STATE_NINTEGER .OR. HTIME.NE.ZC_STATE_NTIME
     D .OR. HTAPE.NE.ZC_TAPE_WIDTH) THEN
         WRITE(*,*) 'ERROR: invalid path-file header'
         STOP 2
      ENDIF
      READ(200,IOSTAT=IOS) STATE_R,STATE_C,STATE_D,STATE_I,
     A STATE_TIME,TAPE
      IF(IOS.NE.0) THEN
         WRITE(*,*) 'ERROR: cannot read path payload; IOSTAT=',IOS
         STOP 2
      ENDIF
      CLOSE(200)
      RETURN
      END


C     These two helpers move passive scratch arrays in the generated
C     wrapper.  They are deliberately not differentiated.
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
