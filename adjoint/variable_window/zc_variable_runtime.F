C=======================================================================
C Shared runtime and stream helpers for the isolated variable-window
C drivers.  These routines reproduce the run23 final3 static setup but do
C not contain or modify any scientific kernel implementation.
C=======================================================================
      SUBROUTINE ZC_VW_INITIALIZE
      INCLUDE 'zc_kernel_state.inc'
      INCLUDE 'zeq.common'
      INCLUDE 'modified_means.common'

      INTEGER NSTART,NT_RESTART,NATMR,IT1,IOS
      REAL TFIND,TD_RESTART,TZERO_RESTART
      LOGICAL ZC_KERNEL_MODE
      COMMON/commTIME/IT,MP,TY
      COMMON/IST/ist
      COMMON/STRPAR/RSTR,TCUT,TSEED,TAMP1,TAMP2,JLEFT,JRT,ITOP,
     A IBOT,NIC
      COMMON/zc_kernel_mode/ZC_KERNEL_MODE
      CHARACTER*1 HEADER_LOCAL(80)

      ZC_KERNEL_MODE=.TRUE.
      iprint_namelists=0
      ist=0
      OPEN(5,FILE='fc.data',STATUS='OLD',IOSTAT=IOS)
      IF(IOS.NE.0) THEN
         WRITE(*,*) 'ERROR: cannot open fc.data; IOSTAT=',IOS
         STOP 2
      ENDIF
      READ(5,450,IOSTAT=IOS) HEADER_LOCAL
 450  FORMAT(80A1)
      IF(IOS.NE.0) THEN
         WRITE(*,*) 'ERROR: cannot read fc.data header; IOSTAT=',IOS
         STOP 2
      ENDIF
      CALL OPENFL
      NATM=2
      READ(5,40,IOSTAT=IOS) NSTART,TFIND
 40   FORMAT(8X,I15/8X,F15.0)
      IF(IOS.NE.0) THEN
         WRITE(*,*) 'ERROR: cannot read restart configuration; IOSTAT=',IOS
         STOP 2
      ENDIF
      IF(NSTART.NE.3) THEN
         WRITE(*,*) 'ERROR: variable-window driver requires NSTART=3'
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
         WRITE(*,*) 'ERROR: variable-window driver requires NIC=0'
         STOP 2
      ENDIF
      RETURN
      END


      SUBROUTINE ZC_VW_PARSE_STEPS(TEXT,NSTEPS)
      CHARACTER*(*) TEXT
      INTEGER NSTEPS,IOS,MAX_STEPS
      PARAMETER (MAX_STEPS=10000)
      READ(TEXT,*,IOSTAT=IOS) NSTEPS
      IF(IOS.NE.0 .OR. NSTEPS.LT.1 .OR. NSTEPS.GT.MAX_STEPS) THEN
         WRITE(*,*) 'ERROR: steps must be an integer in [1,10000]'
         STOP 2
      ENDIF
      RETURN
      END


      SUBROUTINE ZC_VW_READ_STATE(NAME,STATE_R,STATE_C,STATE_D,
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
      INQUIRE(FILE=NAME,EXIST=EXISTS)
      IF(.NOT.EXISTS) THEN
         WRITE(*,*) 'ERROR: state file does not exist: ',TRIM(NAME)
         STOP 2
      ENDIF
      INQUIRE(FILE=NAME,SIZE=FILE_BYTES)
      IF(FILE_BYTES.NE.EXPECTED_BYTES) THEN
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


      SUBROUTINE ZC_VW_READ_REAL(NAME,X,N)
      CHARACTER*(*) NAME
      REAL X(N)
      INTEGER N,IOS,I
      INTEGER*8 FILE_BYTES,EXPECTED_BYTES
      LOGICAL EXISTS
      EXPECTED_BYTES=4_8*N
      INQUIRE(FILE=NAME,EXIST=EXISTS)
      IF(.NOT.EXISTS) THEN
         WRITE(*,*) 'ERROR: real-vector file does not exist: ',TRIM(NAME)
         STOP 2
      ENDIF
      INQUIRE(FILE=NAME,SIZE=FILE_BYTES)
      IF(FILE_BYTES.NE.EXPECTED_BYTES) THEN
         WRITE(*,*) 'ERROR: invalid real-vector bytes: ',FILE_BYTES,
     A    ' expected ',EXPECTED_BYTES
         STOP 2
      ENDIF
      OPEN(201,FILE=NAME,FORM='UNFORMATTED',ACCESS='STREAM',
     A STATUS='OLD',IOSTAT=IOS)
      IF(IOS.NE.0) THEN
         WRITE(*,*) 'ERROR: cannot open real vector; IOSTAT=',IOS
         STOP 2
      ENDIF
      READ(201,IOSTAT=IOS) X
      IF(IOS.NE.0) THEN
         WRITE(*,*) 'ERROR: cannot read real vector; IOSTAT=',IOS
         STOP 2
      ENDIF
      CLOSE(201)
      DO 10 I=1,N
         IF(X(I).NE.X(I) .OR. ABS(X(I)).GT.HUGE(X(I))) THEN
            WRITE(*,*) 'ERROR: real vector contains nonfinite values'
            STOP 2
         ENDIF
 10   CONTINUE
      RETURN
      END


      SUBROUTINE ZC_VW_WRITE_REAL(NAME,X,N)
      CHARACTER*(*) NAME
      REAL X(N)
      INTEGER N,IOS
      OPEN(202,FILE=NAME,FORM='UNFORMATTED',ACCESS='STREAM',
     A STATUS='REPLACE',IOSTAT=IOS)
      IF(IOS.NE.0) THEN
         WRITE(*,*) 'ERROR: cannot open real-vector output; IOSTAT=',IOS
         STOP 2
      ENDIF
      WRITE(202,IOSTAT=IOS) X
      IF(IOS.NE.0) THEN
         WRITE(*,*) 'ERROR: cannot write real vector; IOSTAT=',IOS
         STOP 2
      ENDIF
      CLOSE(202)
      RETURN
      END


      SUBROUTINE ZC_VW_READ_PATH_HEADER(NAME,NSTEPS)
      INCLUDE 'zc_kernel_state.inc'
      CHARACTER*(*) NAME
      INTEGER NSTEPS,HREAL,HCOMPLEX,HDOUBLE,HINTEGER,HTIME,HTAPE,IOS
      INTEGER MAX_STEPS
      PARAMETER (MAX_STEPS=10000)
      INTEGER*8 FILE_BYTES,EXPECTED_BYTES
      LOGICAL EXISTS
      INQUIRE(FILE=NAME,EXIST=EXISTS)
      IF(.NOT.EXISTS) THEN
         WRITE(*,*) 'ERROR: path file does not exist: ',TRIM(NAME)
         STOP 2
      ENDIF
      OPEN(203,FILE=NAME,FORM='UNFORMATTED',ACCESS='STREAM',
     A STATUS='OLD',IOSTAT=IOS)
      IF(IOS.NE.0) THEN
         WRITE(*,*) 'ERROR: cannot open path file; IOSTAT=',IOS
         STOP 2
      ENDIF
      READ(203,IOSTAT=IOS) NSTEPS,HREAL,HCOMPLEX,HDOUBLE,HINTEGER,
     A HTIME,HTAPE
      CLOSE(203)
      IF(IOS.NE.0) THEN
         WRITE(*,*) 'ERROR: cannot read path header; IOSTAT=',IOS
         STOP 2
      ENDIF
      IF(NSTEPS.LT.1 .OR. NSTEPS.GT.MAX_STEPS .OR.
     A HREAL.NE.ZC_STATE_NREAL .OR.
     B HCOMPLEX.NE.ZC_STATE_NCOMPLEX .OR.
     C HDOUBLE.NE.ZC_STATE_NDOUBLE .OR.
     D HINTEGER.NE.ZC_STATE_NINTEGER .OR. HTIME.NE.ZC_STATE_NTIME
     E .OR. HTAPE.NE.ZC_TAPE_WIDTH) THEN
         WRITE(*,*) 'ERROR: invalid variable-window path header'
         STOP 2
      ENDIF
      EXPECTED_BYTES=7_8*4_8+
     A (NSTEPS+1_8)*(4_8*ZC_STATE_NREAL+
     B  8_8*ZC_STATE_NCOMPLEX+8_8*ZC_STATE_NDOUBLE+
     C  4_8*ZC_STATE_NINTEGER+4_8*ZC_STATE_NTIME)+
     D NSTEPS*4_8*ZC_TAPE_WIDTH
      INQUIRE(FILE=NAME,SIZE=FILE_BYTES)
      IF(FILE_BYTES.NE.EXPECTED_BYTES) THEN
         WRITE(*,*) 'ERROR: invalid path-file bytes: ',FILE_BYTES,
     A    ' expected ',EXPECTED_BYTES
         STOP 2
      ENDIF
      RETURN
      END


      SUBROUTINE ZC_VW_READ_PATH(NAME,NSTEPS,STATE_R,STATE_C,
     A STATE_D,STATE_I,STATE_TIME,TAPE)
      INCLUDE 'zc_kernel_state.inc'
      CHARACTER*(*) NAME
      INTEGER NSTEPS
      REAL STATE_R(ZC_STATE_NREAL,0:NSTEPS)
      COMPLEX STATE_C(ZC_STATE_NCOMPLEX,0:NSTEPS)
      REAL*8 STATE_D(ZC_STATE_NDOUBLE,0:NSTEPS)
      INTEGER STATE_I(ZC_STATE_NINTEGER,0:NSTEPS)
      REAL STATE_TIME(ZC_STATE_NTIME,0:NSTEPS)
      INTEGER TAPE(ZC_TAPE_WIDTH,NSTEPS)
      INTEGER HSTEPS,HREAL,HCOMPLEX,HDOUBLE,HINTEGER,HTIME,HTAPE,IOS
      CALL ZC_VW_READ_PATH_HEADER(NAME,HSTEPS)
      IF(HSTEPS.NE.NSTEPS) THEN
         WRITE(*,*) 'ERROR: path changed between header and payload reads'
         STOP 2
      ENDIF
      OPEN(203,FILE=NAME,FORM='UNFORMATTED',ACCESS='STREAM',
     A STATUS='OLD',IOSTAT=IOS)
      READ(203,IOSTAT=IOS) HSTEPS,HREAL,HCOMPLEX,HDOUBLE,HINTEGER,
     A HTIME,HTAPE
      IF(IOS.EQ.0) READ(203,IOSTAT=IOS) STATE_R,STATE_C,STATE_D,
     A STATE_I,STATE_TIME,TAPE
      CLOSE(203)
      IF(IOS.NE.0) THEN
         WRITE(*,*) 'ERROR: cannot read path payload; IOSTAT=',IOS
         STOP 2
      ENDIF
      RETURN
      END


      SUBROUTINE ZC_VW_WRITE_PATH(NAME,NSTEPS,STATE_R,STATE_C,
     A STATE_D,STATE_I,STATE_TIME,TAPE)
      INCLUDE 'zc_kernel_state.inc'
      CHARACTER*(*) NAME
      INTEGER NSTEPS
      REAL STATE_R(ZC_STATE_NREAL,0:NSTEPS)
      COMPLEX STATE_C(ZC_STATE_NCOMPLEX,0:NSTEPS)
      REAL*8 STATE_D(ZC_STATE_NDOUBLE,0:NSTEPS)
      INTEGER STATE_I(ZC_STATE_NINTEGER,0:NSTEPS)
      REAL STATE_TIME(ZC_STATE_NTIME,0:NSTEPS)
      INTEGER TAPE(ZC_TAPE_WIDTH,NSTEPS)
      INTEGER IOS
      OPEN(204,FILE=NAME,FORM='UNFORMATTED',ACCESS='STREAM',
     A STATUS='REPLACE',IOSTAT=IOS)
      IF(IOS.NE.0) THEN
         WRITE(*,*) 'ERROR: cannot open path output; IOSTAT=',IOS
         STOP 2
      ENDIF
      WRITE(204,IOSTAT=IOS) NSTEPS,ZC_STATE_NREAL,ZC_STATE_NCOMPLEX,
     A ZC_STATE_NDOUBLE,ZC_STATE_NINTEGER,ZC_STATE_NTIME,ZC_TAPE_WIDTH
      IF(IOS.EQ.0) WRITE(204,IOSTAT=IOS) STATE_R,STATE_C,STATE_D,
     A STATE_I,STATE_TIME,TAPE
      CLOSE(204)
      IF(IOS.NE.0) THEN
         WRITE(*,*) 'ERROR: cannot write path output; IOSTAT=',IOS
         STOP 2
      ENDIF
      RETURN
      END
