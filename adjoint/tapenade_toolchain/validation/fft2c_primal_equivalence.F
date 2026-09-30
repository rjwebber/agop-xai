C=======================================================================
C Bitwise comparison of authentic FFT2C and the AD-ready source rewrite.
C The reference source is compiled with the preprocessor rename
C FFT2C=FFT2C_REF so that both implementations can be linked here.
C=======================================================================
      PROGRAM TEST_FFT2C_PRIMAL_EQUIVALENCE
      IMPLICIT NONE
      INTEGER N,M,NIWK,I,FAILURES
      PARAMETER (N=64,M=6,NIWK=534)
      INTEGER IWK1(NIWK),IWK2(NIWK)
      COMPLEX*16 A(N),B(N)

      DO 10 I=1,N
         A(I)=DCMPLX(DSIN(0.17D0*I)+0.01D0*I,
     A              DCOS(0.11D0*I)-0.007D0*I)
         B(I)=A(I)
 10   CONTINUE
      CALL FFT2C_REF(A,N,M,IWK1,NIWK)
      CALL FFT2C(B,N,M,IWK2,NIWK)
      FAILURES=0
      DO 20 I=1,N
         IF(A(I).NE.B(I)) FAILURES=FAILURES+1
 20   CONTINUE
      WRITE(*,'(A,I0)') 'bitwise_mismatches=',FAILURES
      IF(FAILURES.NE.0) THEN
         WRITE(*,'(A)') 'FAIL'
         STOP 1
      ENDIF
      WRITE(*,'(A)') 'PASS'
      END
