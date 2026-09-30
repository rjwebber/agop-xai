C=======================================================================
C Scalar 31-transition head for automatic differentiation of the
C certified explicit-state ZC kernel.  The input checkpoint immediately
C precedes the neural-network input update; transition 31 produces the
C matching ten-month target state.  The complete continuous real state,
C including the carried Kelvin-amplitude work vector AK, is active.
C Complex FFT workspace is overwritten before every read;
C discrete clock, branch, and disabled stochastic-forcing values are passive.
C=======================================================================
      SUBROUTINE ZC_KERNEL_NINO3(STATE_R_IN,STATE_C_IN,STATE_D_IN,
     A STATE_I_IN,STATE_TIME_IN,NINO3)
      INCLUDE 'zc_kernel_state.inc'
      INTEGER TARGET_STEPS,TO_OFFSET
      PARAMETER (TARGET_STEPS=31,TO_OFFSET=32121)
      REAL STATE_R_IN(ZC_STATE_NREAL),STATE_R_OUT(ZC_STATE_NREAL)
      COMPLEX STATE_C_IN(ZC_STATE_NCOMPLEX)
      COMPLEX STATE_C_OUT(ZC_STATE_NCOMPLEX)
      REAL*8 STATE_D_IN(ZC_STATE_NDOUBLE)
      REAL*8 STATE_D_OUT(ZC_STATE_NDOUBLE)
      INTEGER STATE_I_IN(ZC_STATE_NINTEGER)
      INTEGER STATE_I_OUT(ZC_STATE_NINTEGER)
      REAL STATE_TIME_IN(ZC_STATE_NTIME),STATE_TIME_OUT(ZC_STATE_NTIME)
      INTEGER TAPE(ZC_TAPE_WIDTH,TARGET_STEPS)
      REAL NINO3
      REAL*8 NINO3_SUM

      CALL ZC_KERNEL_WINDOW(STATE_R_IN,STATE_C_IN,STATE_D_IN,
     A STATE_I_IN,STATE_TIME_IN,TARGET_STEPS,STATE_R_OUT,STATE_C_OUT,
     B STATE_D_OUT,STATE_I_OUT,STATE_TIME_OUT,TAPE)

C     TO is packed in Fortran column-major order.  Its zero-based segment
C     offset is 32121.  The canonical zc-v3 target uses I=13:18,J=21:31;
C     do not confuse this with the atmosphere reset's legacy J=20:30 box.
      NINO3_SUM=0.0D0
      DO 20 J=21,31
         DO 10 I=13,18
            K=TO_OFFSET+I+30*(J-1)
            NINO3_SUM=NINO3_SUM+DBLE(STATE_R_OUT(K))
 10      CONTINUE
 20   CONTINUE
      NINO3=NINO3_SUM/66.0D0
      RETURN
      END
