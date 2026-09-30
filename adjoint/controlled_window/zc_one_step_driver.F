C=======================================================================
C Advance one complete explicit ZC state by one authentic coupled step.
C
C This driver is intentionally separate from the canonical run23 evidence
C executables.  It provides the small compositional primitive needed for
C externally applied, covariance-balanced incremental controls.  The active
C kernel and all immutable setup objects are reused without modification.
C=======================================================================
      PROGRAM ZC_ONE_STEP_DRIVER
      INCLUDE 'zc_kernel_state.inc'
      INCLUDE 'zeq.common'
      INCLUDE 'modified_means.common'

      REAL STATE_R_IN(ZC_STATE_NREAL),STATE_R_OUT(ZC_STATE_NREAL)
      COMPLEX STATE_C_IN(ZC_STATE_NCOMPLEX)
      COMPLEX STATE_C_OUT(ZC_STATE_NCOMPLEX)
      REAL*8 STATE_D_IN(ZC_STATE_NDOUBLE),STATE_D_OUT(ZC_STATE_NDOUBLE)
      INTEGER STATE_I_IN(ZC_STATE_NINTEGER)
      INTEGER STATE_I_OUT(ZC_STATE_NINTEGER)
      REAL STATE_TIME_IN(ZC_STATE_NTIME),STATE_TIME_OUT(ZC_STATE_NTIME)
      INTEGER TAPE(ZC_TAPE_WIDTH,1)
      INTEGER NSTART,NT_RESTART,NATMR,IT1,IOS
      REAL TFIND,TD_RESTART,TZERO_RESTART
      LOGICAL ZC_KERNEL_MODE
      COMMON/commTIME/IT,MP,TY
      COMMON/IST/ist
      COMMON/STRPAR/RSTR,TCUT,TSEED,TAMP1,TAMP2,JLEFT,JRT,ITOP,
     A IBOT,NIC
      COMMON/zc_kernel_mode/ZC_KERNEL_MODE
      CHARACTER*1 HEADER_LOCAL(80)
      CHARACTER*512 INPUT_FILE,OUTPUT_FILE,TAPE_FILE

      INPUT_FILE=' '
      OUTPUT_FILE=' '
      TAPE_FILE=' '
      CALL GET_COMMAND_ARGUMENT(1,INPUT_FILE)
      CALL GET_COMMAND_ARGUMENT(2,OUTPUT_FILE)
      CALL GET_COMMAND_ARGUMENT(3,TAPE_FILE)
      IF(LEN_TRIM(INPUT_FILE).EQ.0) INPUT_FILE='kernel_input_state.bin'
      IF(LEN_TRIM(OUTPUT_FILE).EQ.0) OUTPUT_FILE='kernel_output_state.bin'
      IF(LEN_TRIM(TAPE_FILE).EQ.0) TAPE_FILE='kernel_one_step_tape.bin'

C     Rebuild the same immutable/static context as the run23 kernel.
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
         WRITE(*,*) 'ERROR: one-step driver requires NSTART=3'
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
         WRITE(*,*) 'ERROR: one-step driver requires deterministic NIC=0'
         STOP 2
      ENDIF

      CALL ZC_CW_READ_STATE(TRIM(INPUT_FILE),STATE_R_IN,STATE_C_IN,
     A STATE_D_IN,STATE_I_IN,STATE_TIME_IN)
      CALL ZC_KERNEL_WINDOW(STATE_R_IN,STATE_C_IN,STATE_D_IN,STATE_I_IN,
     A STATE_TIME_IN,1,STATE_R_OUT,STATE_C_OUT,STATE_D_OUT,STATE_I_OUT,
     B STATE_TIME_OUT,TAPE)
      CALL ZC_CW_WRITE_STATE(TRIM(OUTPUT_FILE),STATE_R_OUT,STATE_C_OUT,
     A STATE_D_OUT,STATE_I_OUT,STATE_TIME_OUT)
      OPEN(202,FILE=TRIM(TAPE_FILE),FORM='UNFORMATTED',
     A ACCESS='STREAM',STATUS='REPLACE')
      WRITE(202) TAPE
      CLOSE(202)
      STOP
 999  WRITE(*,*) 'ERROR: unexpected end of fc.data'
      STOP 2
      END


      SUBROUTINE ZC_CW_READ_STATE(NAME,STATE_R,STATE_C,STATE_D,
     A STATE_I,STATE_TIME)
      INCLUDE 'zc_kernel_state.inc'
      CHARACTER*(*) NAME
      REAL STATE_R(ZC_STATE_NREAL)
      COMPLEX STATE_C(ZC_STATE_NCOMPLEX)
      REAL*8 STATE_D(ZC_STATE_NDOUBLE)
      INTEGER STATE_I(ZC_STATE_NINTEGER)
      REAL STATE_TIME(ZC_STATE_NTIME)
      INTEGER IOS
      INTEGER*8 FILE_BYTES,EXPECTED_BYTES
      LOGICAL EXISTS
      EXPECTED_BYTES=4_8*ZC_STATE_NREAL+8_8*ZC_STATE_NCOMPLEX+
     A 8_8*ZC_STATE_NDOUBLE+4_8*ZC_STATE_NINTEGER+
     B 4_8*ZC_STATE_NTIME
      INQUIRE(FILE=NAME,EXIST=EXISTS,SIZE=FILE_BYTES)
      IF(.NOT.EXISTS .OR. FILE_BYTES.NE.EXPECTED_BYTES) THEN
         WRITE(*,*) 'ERROR: invalid state-file bytes: ',FILE_BYTES,
     A    ' expected ',EXPECTED_BYTES
         STOP 2
      ENDIF
      OPEN(200,FILE=NAME,FORM='UNFORMATTED',ACCESS='STREAM',
     A STATUS='OLD',IOSTAT=IOS)
      IF(IOS.NE.0) THEN
         WRITE(*,*) 'ERROR: cannot open state file; IOSTAT=',IOS
         STOP 2
      ENDIF
      READ(200,IOSTAT=IOS) STATE_R,STATE_C,STATE_D,STATE_I,STATE_TIME
      IF(IOS.NE.0) THEN
         WRITE(*,*) 'ERROR: cannot read state file; IOSTAT=',IOS
         STOP 2
      ENDIF
      CLOSE(200)
      RETURN
      END


      SUBROUTINE ZC_CW_WRITE_STATE(NAME,STATE_R,STATE_C,STATE_D,
     A STATE_I,STATE_TIME)
      INCLUDE 'zc_kernel_state.inc'
      CHARACTER*(*) NAME
      REAL STATE_R(ZC_STATE_NREAL)
      COMPLEX STATE_C(ZC_STATE_NCOMPLEX)
      REAL*8 STATE_D(ZC_STATE_NDOUBLE)
      INTEGER STATE_I(ZC_STATE_NINTEGER)
      REAL STATE_TIME(ZC_STATE_NTIME)
      OPEN(201,FILE=NAME,FORM='UNFORMATTED',ACCESS='STREAM',
     A STATUS='REPLACE')
      WRITE(201) STATE_R,STATE_C,STATE_D,STATE_I,STATE_TIME
      CLOSE(201)
      RETURN
      END
