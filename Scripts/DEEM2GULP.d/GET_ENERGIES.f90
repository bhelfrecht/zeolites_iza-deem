! Giovanni Pireddu, 2019
PROGRAM GE
  IMPLICIT NONE
  CHARACTER(LEN=2) :: Lab
  CHARACTER(LEN= 100) :: COMMAND
  CHARACTER(LEN= 20) :: FILENAME
  CHARACTER(LEN= 50) :: AW
  CHARACTER(LEN= 42) :: WW
  CHARACTER(LEN= 20), DIMENSION(:) :: List2(2), List3(3), List4(4), List5(5), List6(6), List7(7)
  INTEGER :: Count, ifile, iRead, SiCount, idum, iline, num
  REAL(KIND= 8) :: IntPot, TPot, MMReal, MMRecip, MMTotal, TotEn
  LOGICAL :: LCheck

  COMMAND= 'ls log_GULP_*.out > Files.txt'

  CALL SYSTEM( COMMAND )

  OPEN( UNIT= 100, FILE= 'Files.txt', ACTION= 'READ' )
  OPEN( UNIT= 101, FILE= 'Energies.out', ACTION= 'WRITE' )
  OPEN( UNIT= 121, FILE= 'Rogues.out', ACTION= 'WRITE' )

  WRITE(101,*) '# id / Name / nSi / InteractPot / ThreeBodyPot / M-M(Real) / M-M(Recip) / M-M(Total) / TotalEn'

  ifile= 0
  iline= 0
  DO 
     READ(100,*,END=10) FILENAME
     ifile= ifile + 1
     OPEN( UNIT= 200, FILE= FILENAME, ACTION= 'READ' )
     Count= 0
     LCheck= .FALSE.
     DO 
        READ(200,*,END=20) AW
        LCheck= .TRUE.
        IF( AW == 'Formula' ) THEN
           READ(200,*) 
           READ(200,'(A,I5)') WW, num
           Sicount= num / 5
        END IF

        IF( AW == 'Components' ) THEN
           Count= Count + 1
           IF( Count == 2 ) THEN
              READ(200,*)
              READ(200,*)
              READ(200,*) List3(:), IntPot
              READ(200,*) List3(:), TPot
              READ(200,*) List5(:), MMReal
              READ(200,*) List4(:), MMRecip
              READ(200,*) List4(:), MMTotal
              READ(200,*)
              READ(200,*) List4(:)
              IF( List4(4) == '=' ) THEN
                 BACKSPACE( 200 )
                 READ(200,*) List4(:), TotEn
              ELSE
                 READ(200,*) List4(:), TotEn
              END IF

              WRITE(*,*) ifile, TotEn, SiCount

              !Convert to kJ/mol
              IntPot= IntPot * 96.485307
              TPot= TPot * 96.485307
              MMReal= MMReal * 96.485307
              MMRecip= MMRecip * 96.485307
              MMTotal= MMTotal * 96.485307
              TotEn= TotEn * 96.485307

              IntPot= IntPot / DBLE( SiCount )
              TPot= TPot / DBLE( SiCount )
              MMReal= MMReal / DBLE( SiCount )
              MMRecip= MMRecip / DBLE( SiCount )
              MMTotal= MMTotal / DBLE( SiCount )
              TotEn= TotEn / DBLE( SiCount )

              !Print results
              iline= iline + 1

              WRITE(101,*) ifile, FILENAME(10:16), SiCount &
                   & , IntPot, TPot, MMReal, MMRecip, MMTotal &
                   & , TotEn

           END IF
        END IF
     END DO

20   CONTINUE

     IF( Count /= 2 ) THEN 
        WRITE(*,*) 'Something strange is happening, overcounting / undercounting "Components"', Count, FILENAME
        WRITE(121,*) FILENAME(10:16)
        STOP
     END IF

     IF( .NOT. LCheck ) THEN
        STOP
     END IF

  END DO

10 CONTINUE

  CLOSE( 100 )

END PROGRAM GE
