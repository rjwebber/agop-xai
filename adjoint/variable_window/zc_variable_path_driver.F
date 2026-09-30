C=======================================================================
C Runtime-configurable authentic-primal boundary path.  This is a thin
C driver over the unchanged run23 final3 ZC_KERNEL_WINDOW object code.
C
C Arguments: initial-state path, output-path path, number of transitions.
C The default transition count is 31 for compatibility with final3.
C=======================================================================
      PROGRAM ZC_VARIABLE_PATH_DRIVER
      INCLUDE 'zc_kernel_state.inc'

      REAL STATE_R_IN(ZC_STATE_NREAL)
      COMPLEX STATE_C_IN(ZC_STATE_NCOMPLEX)
      REAL*8 STATE_D_IN(ZC_STATE_NDOUBLE)
      INTEGER STATE_I_IN(ZC_STATE_NINTEGER)
      REAL STATE_TIME_IN(ZC_STATE_NTIME)
      REAL, ALLOCATABLE :: STATE_R_PATH(:,:)
      COMPLEX, ALLOCATABLE :: STATE_C_PATH(:,:)
      REAL*8, ALLOCATABLE :: STATE_D_PATH(:,:)
      INTEGER, ALLOCATABLE :: STATE_I_PATH(:,:)
      REAL, ALLOCATABLE :: STATE_TIME_PATH(:,:)
      INTEGER, ALLOCATABLE :: TAPE_REFERENCE(:,:)
      INTEGER TO_OFFSET
      PARAMETER (TO_OFFSET=32121)
      REAL NINO3,TSTART,TEND
      DOUBLE PRECISION NINO3_SUM
      INTEGER I,J,K,N,NSTEPS,IOS
      CHARACTER*512 STATE_FILE,PATH_FILE,META_FILE,STEPS_TEXT

      STATE_FILE=' '
      PATH_FILE=' '
      STEPS_TEXT=' '
      CALL GET_COMMAND_ARGUMENT(1,STATE_FILE)
      CALL GET_COMMAND_ARGUMENT(2,PATH_FILE)
      CALL GET_COMMAND_ARGUMENT(3,STEPS_TEXT)
      IF(LEN_TRIM(STATE_FILE).EQ.0) STATE_FILE='kernel_initial_state.bin'
      IF(LEN_TRIM(PATH_FILE).EQ.0) PATH_FILE='zc_variable_path.bin'
      NSTEPS=31
      IF(LEN_TRIM(STEPS_TEXT).GT.0) THEN
         CALL ZC_VW_PARSE_STEPS(TRIM(STEPS_TEXT),NSTEPS)
      ENDIF

      CALL ZC_VW_INITIALIZE
      CALL ZC_VW_READ_STATE(TRIM(STATE_FILE),STATE_R_IN,STATE_C_IN,
     A STATE_D_IN,STATE_I_IN,STATE_TIME_IN)
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

      DO 10 I=1,ZC_STATE_NREAL
         STATE_R_PATH(I,0)=STATE_R_IN(I)
 10   CONTINUE
      DO 11 I=1,ZC_STATE_NCOMPLEX
         STATE_C_PATH(I,0)=STATE_C_IN(I)
 11   CONTINUE
      DO 12 I=1,ZC_STATE_NDOUBLE
         STATE_D_PATH(I,0)=STATE_D_IN(I)
 12   CONTINUE
      DO 13 I=1,ZC_STATE_NINTEGER
         STATE_I_PATH(I,0)=STATE_I_IN(I)
 13   CONTINUE
      DO 14 I=1,ZC_STATE_NTIME
         STATE_TIME_PATH(I,0)=STATE_TIME_IN(I)
 14   CONTINUE

      CALL CPU_TIME(TSTART)
      DO 20 N=1,NSTEPS
         CALL ZC_KERNEL_WINDOW(STATE_R_PATH(1,N-1),
     A    STATE_C_PATH(1,N-1),STATE_D_PATH(1,N-1),
     B    STATE_I_PATH(1,N-1),STATE_TIME_PATH(1,N-1),1,
     C    STATE_R_PATH(1,N),STATE_C_PATH(1,N),STATE_D_PATH(1,N),
     D    STATE_I_PATH(1,N),STATE_TIME_PATH(1,N),
     E    TAPE_REFERENCE(1,N))
 20   CONTINUE
      CALL CPU_TIME(TEND)

      NINO3_SUM=0.0D0
      DO 30 J=21,31
         DO 25 I=13,18
            K=TO_OFFSET+I+30*(J-1)
            NINO3_SUM=NINO3_SUM+DBLE(STATE_R_PATH(K,NSTEPS))
 25      CONTINUE
 30   CONTINUE
      NINO3=NINO3_SUM/66.0D0

      CALL ZC_VW_WRITE_PATH(TRIM(PATH_FILE),NSTEPS,STATE_R_PATH,
     A STATE_C_PATH,STATE_D_PATH,STATE_I_PATH,STATE_TIME_PATH,
     B TAPE_REFERENCE)
      META_FILE=TRIM(PATH_FILE)//'.txt'
      OPEN(202,FILE=TRIM(META_FILE),STATUS='REPLACE',IOSTAT=IOS)
      IF(IOS.NE.0) STOP 'cannot open path metadata output'
      WRITE(202,'(A)') 'producer=normalized authentic primal kernel'
      WRITE(202,'(A)') 'driver=isolated runtime-variable window'
      WRITE(202,'(A)') 'state_semantics=pre-input checkpoint'
      WRITE(202,'(A,I0)') 'transitions=',NSTEPS
      WRITE(202,'(A)') 'fortran_box=I=13:18,J=21:31'
      WRITE(202,'(A,ES24.16)') 'final_nino3_celsius=',NINO3
      WRITE(202,'(A,ES24.16)') 'path_cpu_seconds=',TEND-TSTART
      CLOSE(202)
      WRITE(*,'(A,I0)') 'Transitions: ',NSTEPS
      WRITE(*,'(A,ES16.8)') 'Final canonical Nino-3 (C): ',NINO3
      WRITE(*,'(A,A)') 'Authentic primal path: ',TRIM(PATH_FILE)
      STOP
      END
