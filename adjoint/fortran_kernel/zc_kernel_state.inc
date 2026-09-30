C Fixed-layout state contract for the deterministic ZC kernel.
C
C The real state contains every REAL value written in numeric restart
C records 2--4.  TD and TY are carried once in the separate passive clock
C vector.  Complex atmospheric memory and the double-precision random-
C generator state retain their native types.  Discrete clock/branch
C variables are carried separately and are not active differentiation
C variables.
      INTEGER ZC_STATE_NREAL,ZC_STATE_NCOMPLEX,ZC_STATE_NDOUBLE
      INTEGER ZC_STATE_NINTEGER,ZC_STATE_NTIME,ZC_TAPE_WIDTH
C 59,069 values reproduce restart records 2--4.  The final 79 values are
C AK(1:NXP), an omitted but genuine Kelvin-wave memory read at the western
C boundary before it is rebuilt.  Padding, scratch U/H/F/G/BK, and the
C passive clock remain excluded.
      PARAMETER (ZC_STATE_NREAL=59148)
      PARAMETER (ZC_STATE_NCOMPLEX=5280)
      PARAMETER (ZC_STATE_NDOUBLE=1)
      PARAMETER (ZC_STATE_NINTEGER=4)
      PARAMETER (ZC_STATE_NTIME=2)
      PARAMETER (ZC_TAPE_WIDTH=3)
