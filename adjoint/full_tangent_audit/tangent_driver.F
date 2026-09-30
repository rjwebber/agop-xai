C Independent validation driver for the Tapenade-generated tangent-linear
C ZC window.  Runtime setup follows zc_kernel_driver.F, while all derivative
C input and output use raw single-precision stream files.
      PROGRAM ZC_TANGENT_DRIVER
      INCLUDE 'zc_kernel_state.inc'
      INCLUDE 'zeq.common'
      INCLUDE 'modified_means.common'

      REAL STATE_R_IN(ZC_STATE_NREAL),STATE_R_IND(ZC_STATE_NREAL)
      REAL STATE_R_OUT(ZC_STATE_NREAL),STATE_R_OUTD(ZC_STATE_NREAL)
      COMPLEX STATE_C_IN(ZC_STATE_NCOMPLEX)
      COMPLEX STATE_C_OUT(ZC_STATE_NCOMPLEX)
      REAL*8 STATE_D_IN(ZC_STATE_NDOUBLE)
      REAL*8 STATE_D_OUT(ZC_STATE_NDOUBLE)
      INTEGER STATE_I_IN(ZC_STATE_NINTEGER)
      INTEGER STATE_I_OUT(ZC_STATE_NINTEGER)
      REAL STATE_TIME_IN(ZC_STATE_NTIME),STATE_TIME_OUT(ZC_STATE_NTIME)
      INTEGER TAPE(ZC_TAPE_WIDTH,10000)
      LOGICAL ZC_KERNEL_MODE,HAS_KERNEL_INPUT,HAS_NSTEPS
      COMMON/commTIME/IT,MP,TY
      COMMON/IST/ist
      COMMON/zc_kernel_mode/ZC_KERNEL_MODE
      COMMON/STRPAR/RSTR,TCUT,TSEED,TAMP1,TAMP2,JLEFT,JRT,ITOP,IBOT,
     A NIC
      CHARACTER*1 HEADER_LOCAL(80)
      REAL NINO3,NINO3D
      INTEGER IOS

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
         WRITE(*,*) 'ERROR: tangent driver requires NSTART=3'
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
         WRITE(*,*) 'ERROR: tangent driver requires deterministic NIC=0'
         STOP 2
      ENDIF

      INQUIRE(FILE='kernel_input_state.bin',EXIST=HAS_KERNEL_INPUT)
      IF(.NOT.HAS_KERNEL_INPUT) THEN
         WRITE(*,*) 'ERROR: tangent driver requires kernel_input_state.bin'
         STOP 2
      ENDIF
      CALL ZC_READ_STATE_AUDIT('kernel_input_state.bin',STATE_R_IN,
     A STATE_C_IN,STATE_D_IN,STATE_I_IN,STATE_TIME_IN)
      CALL ZC_READ_REAL_AUDIT('tangent_direction.bin',STATE_R_IND,
     A ZC_STATE_NREAL)

      NSTEPS=1
      INQUIRE(FILE='tangent_nsteps.txt',EXIST=HAS_NSTEPS)
      IF(HAS_NSTEPS) THEN
         OPEN(204,FILE='tangent_nsteps.txt',STATUS='OLD',FORM='FORMATTED')
         READ(204,*,IOSTAT=IOS) NSTEPS
         CLOSE(204)
         IF(IOS.NE.0) THEN
            WRITE(*,*) 'ERROR: cannot read tangent_nsteps.txt'
            STOP 2
         ENDIF
      ENDIF
      IF(NSTEPS.LT.1.OR.NSTEPS.GT.10000) THEN
         WRITE(*,*) 'ERROR: invalid tangent window length'
         STOP 2
      ENDIF

      CALL ZC_KERNEL_WINDOW_D(STATE_R_IN,STATE_R_IND,STATE_C_IN,
     A STATE_D_IN,STATE_I_IN,STATE_TIME_IN,NSTEPS,STATE_R_OUT,
     B STATE_R_OUTD,STATE_C_OUT,STATE_D_OUT,STATE_I_OUT,
     C STATE_TIME_OUT,TAPE)
      CALL ZC_WRITE_REAL_AUDIT('tangent_primal_out.bin',STATE_R_OUT,
     A ZC_STATE_NREAL)
      CALL ZC_WRITE_REAL_AUDIT('tangent_jv.bin',STATE_R_OUTD,
     A ZC_STATE_NREAL)
      CALL ZC_WRITE_TAPE_AUDIT('tangent_branch_tape.bin',TAPE,NSTEPS)

C Exercise the separately generated scalar 31-transition head.  This checks
C that the canonical Nino-3 box and its single final division by 66 agree
C with the generic map output used by the independent validator.
      CALL ZC_KERNEL_NINO3_D(STATE_R_IN,STATE_R_IND,STATE_C_IN,
     A STATE_D_IN,STATE_I_IN,STATE_TIME_IN,NINO3,NINO3D)
      OPEN(205,FILE='tangent_nino3.txt',STATUS='REPLACE',FORM='FORMATTED')
      WRITE(205,'(ES24.16,1X,ES24.16)') NINO3,NINO3D
      CLOSE(205)
      CALL CLOSE_FILES
      STOP
 999  WRITE(*,*) 'ERROR: unexpected end of fc.data'
      STOP 2
      END


      SUBROUTINE ZC_READ_STATE_AUDIT(NAME,STATE_R,STATE_C,STATE_D,
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
      READ(200,IOSTAT=IOS) STATE_R,STATE_C,STATE_D,STATE_I,STATE_TIME
      IF(IOS.NE.0) THEN
         WRITE(*,*) 'ERROR: cannot read state file; IOSTAT=',IOS
         STOP 2
      ENDIF
      CLOSE(200)
      RETURN
      END


      SUBROUTINE ZC_READ_REAL_AUDIT(NAME,X,N)
      CHARACTER*(*) NAME
      REAL X(N)
      INTEGER IOS
      INTEGER*8 FILE_BYTES,EXPECTED_BYTES
      LOGICAL EXISTS
      EXPECTED_BYTES=4_8*N
      INQUIRE(FILE=NAME,EXIST=EXISTS,SIZE=FILE_BYTES)
      IF(.NOT.EXISTS .OR. FILE_BYTES.NE.EXPECTED_BYTES) THEN
         WRITE(*,*) 'ERROR: invalid real-vector bytes: ',FILE_BYTES,
     A    ' expected ',EXPECTED_BYTES
         STOP 2
      ENDIF
      OPEN(201,FILE=NAME,FORM='UNFORMATTED',ACCESS='STREAM',
     A STATUS='OLD',IOSTAT=IOS)
      READ(201,IOSTAT=IOS) X
      IF(IOS.NE.0) THEN
         WRITE(*,*) 'ERROR: cannot read real vector; IOSTAT=',IOS
         STOP 2
      ENDIF
      CLOSE(201)
      RETURN
      END


      SUBROUTINE ZC_WRITE_REAL_AUDIT(NAME,X,N)
      CHARACTER*(*) NAME
      REAL X(N)
      OPEN(202,FILE=NAME,FORM='UNFORMATTED',ACCESS='STREAM',
     A STATUS='REPLACE')
      WRITE(202) X
      CLOSE(202)
      RETURN
      END


      SUBROUTINE ZC_WRITE_TAPE_AUDIT(NAME,TAPE,NSTEPS)
      INCLUDE 'zc_kernel_state.inc'
      CHARACTER*(*) NAME
      INTEGER TAPE(ZC_TAPE_WIDTH,*)
      OPEN(203,FILE=NAME,FORM='UNFORMATTED',ACCESS='STREAM',
     A STATUS='REPLACE')
      WRITE(203) ((TAPE(I,N),I=1,ZC_TAPE_WIDTH),N=1,NSTEPS)
      CLOSE(203)
      RETURN
      END


C Passive helpers omitted by Tapenade because the complex FFT workspace is
C not an active input in this validation contract.
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


C GETN selects the discrete SST substep count and is therefore passive in a
C branchwise tangent.  Keep the authentic selector rather than inventing a
C derivative for the integer branch.
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
