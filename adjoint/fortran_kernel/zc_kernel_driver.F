C Test/replay driver for the explicit-state ZC kernel.  Runtime files follow
C the authentic model convention: fc.data, zeq9fsu.hst, Data/, and namelists.
      PROGRAM ZC_KERNEL_DRIVER
      INCLUDE 'zc_kernel_state.inc'
      INCLUDE 'zeq.common'
      INCLUDE 'modified_means.common'

      REAL STATE_R_IN(ZC_STATE_NREAL),STATE_R_OUT(ZC_STATE_NREAL)
      REAL STATE_R_ROUNDTRIP(ZC_STATE_NREAL)
      COMPLEX STATE_C_IN(ZC_STATE_NCOMPLEX)
      COMPLEX STATE_C_OUT(ZC_STATE_NCOMPLEX)
      COMPLEX STATE_C_ROUNDTRIP(ZC_STATE_NCOMPLEX)
      REAL*8 STATE_D_IN(ZC_STATE_NDOUBLE)
      REAL*8 STATE_D_OUT(ZC_STATE_NDOUBLE)
      REAL*8 STATE_D_ROUNDTRIP(ZC_STATE_NDOUBLE)
      INTEGER STATE_I_IN(ZC_STATE_NINTEGER)
      INTEGER STATE_I_OUT(ZC_STATE_NINTEGER)
      INTEGER STATE_I_ROUNDTRIP(ZC_STATE_NINTEGER)
      REAL STATE_TIME_IN(ZC_STATE_NTIME),STATE_TIME_OUT(ZC_STATE_NTIME)
      REAL STATE_TIME_ROUNDTRIP(ZC_STATE_NTIME)
      INTEGER TAPE(ZC_TAPE_WIDTH,10000)
      LOGICAL ZC_KERNEL_MODE
      LOGICAL HAS_KERNEL_INPUT
      LOGICAL HAS_WORKSPACE_POISON
      INTEGER WORKSPACE_POISON_MODE
      COMMON/commTIME/IT,MP,TY
      COMMON/IST/ist
      COMMON/zc_kernel_mode/ZC_KERNEL_MODE
      CHARACTER*1 HEADER_LOCAL(80)

      ZC_KERNEL_MODE=.TRUE.
      iprint_namelists=0
      ist=0
      OPEN(5,FILE='fc.data')
      READ(5,450) HEADER_LOCAL
 450  FORMAT(80A1)
      CALL OPENFL

      NATM=2
      READ(5,40,END=999) NSTART,TFIND
 40   FORMAT(8X,I15/8X,F15.0)
      IF(NSTART.NE.3) THEN
         WRITE(6,'(A)') 'ERROR: kernel driver requires NSTART=3'
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
         WRITE(6,'(A)')
     A    'ERROR: kernel driver requires deterministic NIC=0'
         STOP 2
      ENDIF
      IF(NTIMES.LT.NT) THEN
         WRITE(6,'(A)') 'ERROR: kernel end precedes restart time'
         STOP 2
      ENDIF
      NSTEPS=NTIMES-NT
      IF(NSTEPS.GT.10000) THEN
         WRITE(6,'(A)') 'ERROR: kernel tape allocation exceeded'
         STOP 2
      ENDIF

C Match the authentic restart setup before exposing the explicit state.
      ISTART=0
      CALL STRESS(TD,ISTART)
      CALL CFORCE
      ISTART=1

      CALL ZC_KERNEL_PACK(STATE_R_IN,STATE_C_IN,STATE_D_IN,STATE_I_IN,
     A STATE_TIME_IN)
      INQUIRE(FILE='kernel_input_state.bin',EXIST=HAS_KERNEL_INPUT)
      IF(HAS_KERNEL_INPUT) THEN
         CALL ZC_READ_STATE('kernel_input_state.bin',STATE_R_IN,
     A    STATE_C_IN,STATE_D_IN,STATE_I_IN,STATE_TIME_IN)
         CALL ZC_KERNEL_UNPACK(STATE_R_IN,STATE_C_IN,STATE_D_IN,
     A    STATE_I_IN,STATE_TIME_IN)
      ENDIF
      CALL ZC_WRITE_STATE('kernel_initial_state.bin',STATE_R_IN,
     A STATE_C_IN,STATE_D_IN,STATE_I_IN,STATE_TIME_IN)

C Exercise the inverse map explicitly before the forward calculation.
      CALL ZC_KERNEL_UNPACK(STATE_R_IN,STATE_C_IN,STATE_D_IN,STATE_I_IN,
     A STATE_TIME_IN)
      CALL ZC_KERNEL_PACK(STATE_R_ROUNDTRIP,STATE_C_ROUNDTRIP,
     A STATE_D_ROUNDTRIP,STATE_I_ROUNDTRIP,STATE_TIME_ROUNDTRIP)
      CALL ZC_WRITE_STATE('kernel_roundtrip_state.bin',STATE_R_ROUNDTRIP,
     A STATE_C_ROUNDTRIP,STATE_D_ROUNDTRIP,STATE_I_ROUNDTRIP,
     B STATE_TIME_ROUNDTRIP)

C A test-only sentinel lets the validator overwrite arrays deliberately omitted
C from the explicit state.  Production runs do not create this file.  Any
C dependence on one of these arrays then changes the packed future state.
      INQUIRE(FILE='kernel_workspace_poison.txt',
     A EXIST=HAS_WORKSPACE_POISON)
      IF(HAS_WORKSPACE_POISON) THEN
         OPEN(203,FILE='kernel_workspace_poison.txt',STATUS='OLD',
     A      FORM='FORMATTED')
         READ(203,*) WORKSPACE_POISON_MODE
         CLOSE(203)
         CALL ZC_POISON_EXCLUDED_WORKSPACE(WORKSPACE_POISON_MODE)
         WRITE(6,*) 'ZC omitted-workspace poison mode ',
     A      WORKSPACE_POISON_MODE
      ENDIF

      CALL ZC_KERNEL_WINDOW(STATE_R_IN,STATE_C_IN,STATE_D_IN,STATE_I_IN,
     A STATE_TIME_IN,NSTEPS,STATE_R_OUT,STATE_C_OUT,STATE_D_OUT,
     B STATE_I_OUT,STATE_TIME_OUT,TAPE)
      CALL ZC_WRITE_STATE('kernel_final_state.bin',STATE_R_OUT,
     A STATE_C_OUT,STATE_D_OUT,STATE_I_OUT,STATE_TIME_OUT)
      CALL ZC_WRITE_TAPE('kernel_branch_tape.bin',TAPE,NSTEPS)

C ZC_KERNEL_WINDOW leaves COMMON synchronized with STATE_*_OUT.
      CALL WRHIST
      CALL CLOSE_FILES
      STOP
 999  WRITE(6,'(A)') 'ERROR: unexpected end of fc.data'
      STOP 2
      END


C Test-only adversarial initialization of the ocean arrays omitted from the
C explicit-state contract.  All declared extents are filled, not only the
C active Standard-grid rectangle, so an unexpected boundary or padding read
C is exposed as well.  The values are finite to keep floating-point traps
C useful during debug builds.
      SUBROUTINE ZC_POISON_EXCLUDED_WORKSPACE(MODE)
      INCLUDE 'zeq.common'
      INTEGER MODE

      IF(MODE.EQ.1) THEN
         POISON_SCALE=1.0
      ELSE IF(MODE.EQ.2) THEN
         POISON_SCALE=-1.375
      ELSE
         WRITE(6,'(A)')
     A    'ERROR: workspace poison mode must be 1 or 2'
         STOP 2
      ENDIF
      DO 10 I=1,85
         BK(I)=POISON_SCALE*(11.0+0.001*FLOAT(I))
         DO 20 J=1,117
            U(J,I)=POISON_SCALE*(21.0+0.001*FLOAT(I)
     A         +0.00001*FLOAT(J))
            H(J,I)=POISON_SCALE*(-31.0+0.002*FLOAT(I)
     A         -0.00002*FLOAT(J))
            F(J,I)=POISON_SCALE*(41.0-0.003*FLOAT(I)
     A         +0.00003*FLOAT(J))
            G(J,I)=POISON_SCALE*(-51.0-0.004*FLOAT(I)
     A         -0.00004*FLOAT(J))
 20      CONTINUE
 10   CONTINUE
      RETURN
      END


      SUBROUTINE ZC_WRITE_STATE(NAME,STATE_R,STATE_C,STATE_D,STATE_I,
     A STATE_TIME)
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


      SUBROUTINE ZC_READ_STATE(NAME,STATE_R,STATE_C,STATE_D,STATE_I,
     A STATE_TIME)
      INCLUDE 'zc_kernel_state.inc'
      CHARACTER*(*) NAME
      REAL STATE_R(ZC_STATE_NREAL)
      COMPLEX STATE_C(ZC_STATE_NCOMPLEX)
      REAL*8 STATE_D(ZC_STATE_NDOUBLE)
      INTEGER STATE_I(ZC_STATE_NINTEGER)
      REAL STATE_TIME(ZC_STATE_NTIME)
      INTEGER IOS
      INTEGER*8 FILE_BYTES,EXPECTED_BYTES
      EXPECTED_BYTES=4_8*ZC_STATE_NREAL+8_8*ZC_STATE_NCOMPLEX+
     A 8_8*ZC_STATE_NDOUBLE+4_8*ZC_STATE_NINTEGER+
     B 4_8*ZC_STATE_NTIME
      INQUIRE(FILE=NAME,SIZE=FILE_BYTES,IOSTAT=IOS)
      IF(IOS.NE.0) THEN
         WRITE(6,'(A,A)') 'ERROR: cannot inspect state file: ',NAME
         STOP 2
      ENDIF
      IF(FILE_BYTES.NE.EXPECTED_BYTES) THEN
         WRITE(6,'(A,I0,A,I0)') 'ERROR: state file has ',FILE_BYTES,
     A    ' bytes; expected ',EXPECTED_BYTES
         STOP 2
      ENDIF
      OPEN(200,FILE=NAME,FORM='UNFORMATTED',ACCESS='STREAM',
     A STATUS='OLD',IOSTAT=IOS)
      IF(IOS.NE.0) THEN
         WRITE(6,'(A,A)') 'ERROR: cannot open state file: ',NAME
         STOP 2
      ENDIF
      READ(200,IOSTAT=IOS) STATE_R,STATE_C,STATE_D,STATE_I,STATE_TIME
      IF(IOS.NE.0) THEN
         WRITE(6,'(A,A)') 'ERROR: cannot read state file: ',NAME
         STOP 2
      ENDIF
      CLOSE(200)
      RETURN
      END


      SUBROUTINE ZC_WRITE_TAPE(NAME,TAPE,NSTEPS)
      INCLUDE 'zc_kernel_state.inc'
      CHARACTER*(*) NAME
      INTEGER TAPE(ZC_TAPE_WIDTH,*)
      OPEN(202,FILE=NAME,FORM='UNFORMATTED',ACCESS='STREAM',
     A STATUS='REPLACE')
      IF(NSTEPS.GT.0) WRITE(202) ((TAPE(I,N),I=1,ZC_TAPE_WIDTH),
     A N=1,NSTEPS)
      CLOSE(202)
      RETURN
      END
