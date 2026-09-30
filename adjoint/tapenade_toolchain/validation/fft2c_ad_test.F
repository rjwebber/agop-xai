C=======================================================================
C Independent derivative test for the authentic FFT2C implementation.
C The real Euclidean inner product on complex vectors is used:
C     <X,Y> = Re(sum(conjg(X)*Y)).
C=======================================================================
      PROGRAM TEST_FFT2C_AD
      IMPLICIT NONE
      INTEGER N,M,NIWK,I
      PARAMETER (N=64,M=6,NIWK=534)
      INTEGER IWK(NIWK),IWKD(NIWK),IWKF(NIWK)
      COMPLEX*16 X(N),DIR(N),XD(N),YBAR(N),XT(N),XFD1(N),XFD2(N)
      COMPLEX*16 XR(N),XB(N)
      DOUBLE PRECISION LHS,RHS,FD,EPS,REL,FDREL,SCALE

      DO 10 I=1,N
         X(I)=DCMPLX(DSIN(0.17D0*I)+0.01D0*I,
     A              DCOS(0.11D0*I)-0.007D0*I)
         DIR(I)=DCMPLX(DCOS(0.13D0*I),DSIN(0.19D0*I))
         XD(I)=DIR(I)
         YBAR(I)=DCMPLX(DSIN(0.07D0*I),DCOS(0.23D0*I))
         XT(I)=X(I)
         XFD1(I)=X(I)
         XFD2(I)=X(I)
         XR(I)=X(I)
         XB(I)=YBAR(I)
 10   CONTINUE

C     Tangent transform and its directional inner product.
      CALL FFT2C_D(XT,XD,N,M,IWKD,NIWK)
      LHS=0.0D0
      DO 20 I=1,N
         LHS=LHS+DBLE(DCONJG(YBAR(I))*XD(I))
 20   CONTINUE

C     Reverse transform: forward sweep records the tape, then backward
C     sweep consumes the terminal seed and returns the input cotangent.
      CALL FFT2C_FWD(XR,N,M,IWKF,NIWK)
      CALL FFT2C_BWD(XR,XB,N,M,IWKF,NIWK)
      RHS=0.0D0
      DO 30 I=1,N
         RHS=RHS+DBLE(DCONJG(DIR(I))*XB(I))
 30   CONTINUE

C     Centered finite difference through the untouched primal FFT.
      EPS=1.0D-6
      DO 40 I=1,N
         XFD1(I)=XFD1(I)+EPS*DIR(I)
         XFD2(I)=XFD2(I)-EPS*DIR(I)
 40   CONTINUE
      CALL FFT2C(XFD1,N,M,IWK,NIWK)
      CALL FFT2C(XFD2,N,M,IWK,NIWK)
      FD=0.0D0
      DO 50 I=1,N
         FD=FD+DBLE(DCONJG(YBAR(I))*(XFD1(I)-XFD2(I)))
     A      /(2.0D0*EPS)
 50   CONTINUE

      SCALE=DMAX1(1.0D0,DABS(LHS),DABS(RHS))
      REL=DABS(LHS-RHS)/SCALE
      FDREL=DABS(LHS-FD)/DMAX1(1.0D0,DABS(LHS),DABS(FD))
      WRITE(*,'(A,ES24.16)') 'tangent_dot=',LHS
      WRITE(*,'(A,ES24.16)') 'adjoint_dot=',RHS
      WRITE(*,'(A,ES24.16)') 'centered_fd=',FD
      WRITE(*,'(A,ES24.16)') 'dot_relative_defect=',REL
      WRITE(*,'(A,ES24.16)') 'fd_relative_defect=',FDREL
      IF(REL.GT.1.0D-10.OR.FDREL.GT.1.0D-8) THEN
         WRITE(*,'(A)') 'FAIL'
         STOP 1
      ENDIF
      WRITE(*,'(A)') 'PASS'
      END
