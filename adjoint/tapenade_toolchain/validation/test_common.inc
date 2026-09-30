      INCLUDE 'zeq_d.common'
      INCLUDE 'modified_means_d.common'

C     Reverse variables use Tapenade's generated COMMON layout.  Names such
C     as AKB0 and UB0 are adjoints of AK and U, not primal ZC variables.
      REAL AKB0,AKBB,BKB,HBB,UBB,UB0,HB0,VB,FB,GB
      COMMON /VALS_B/ AKB0(85),AKBB(85),BKB(85),HBB(117,85),
     A UBB(117,85),UB0(117,85),HB0(117,85),VB(117,85),
     B FB(117,85),GB(117,85)
      REAL DB,R1B,R3B,GINTB,TRB,ROSSB,UBNDYB,HBNDYB
      COMMON /ARRAYS_B/ DB(117),R1B(117),R3B(117),GINTB(117),
     A TRB(4),ROSSB(4),UBNDYB(117,4),HBNDYB(117,4)
