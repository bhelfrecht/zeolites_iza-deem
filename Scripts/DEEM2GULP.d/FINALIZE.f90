! Giovanni Pireddu, 2019
PROGRAM FINALIZE
  IMPLICIT NONE
  CHARACTER(LEN=3) :: NAME
  CHARACTER(LEN=20) :: ciffilename, xyzfilename
  CHARACTER(LEN=30) :: AW, W
  CHARACTER(LEN=200) :: NEWCOMMENTLINE
  CHARACTER(LEN=500) :: COMMAND
  INTEGER :: iStruc
  REAL(KIND= 8) :: a, b, c, alpha, beta, gamma, Omega
  REAL(KIND= 8), DIMENSION(1:3) :: aVec, bVec, cVec

  OPEN( UNIT= 100, FILE= 'Names_OPT_xyz.txt', ACTION= 'READ' )
  OPEN( UNIT= 101, FILE= 'Names_OPT_cif.txt', ACTION= 'READ' )

  iStruc= 0
  DO
     READ(100,*,END=10) xyzfilename
     READ(101,*,END=10) ciffilename
     iStruc= iStruc + 1

     OPEN( UNIT= 102, FILE= ciffilename, ACTION= 'READ' )

     DO 
        READ(102,*,END=20) AW

        SELECT CASE( AW )

        CASE( '_cell_length_a')
           BACKSPACE( 102 ) 
           READ(102,*) W, a

        CASE( '_cell_length_b')
           BACKSPACE( 102 ) 
           READ(102,*) W, b

        CASE( '_cell_length_c')
           BACKSPACE( 102 ) 
           READ(102,*) W, c

        CASE( '_cell_angle_alpha')
           BACKSPACE( 102 ) 
           READ(102,*) W, alpha

        CASE( '_cell_angle_beta')
           BACKSPACE( 102 ) 
           READ(102,*) W, beta

        CASE( '_cell_angle_gamma')
           BACKSPACE( 102 ) 
           READ(102,*) W, gamma

        END SELECT

     END DO

20   CONTINUE

     CLOSE( 102 )

     !Generate the lattice vectors

     ! Convert to radians
     alpha= alpha *  3.14159d0 / 180.d0
     beta= beta *  3.14159d0 / 180.d0
     gamma= gamma *  3.14159d0 / 180.d0

     aVec(1)= a
     aVec(2)= 0.d0
     aVec(3)= 0.d0

     bVec(1)= b * COS( gamma )
     bVec(2)= b * SIN( gamma )
     bVec(3)= 0.d0

     cVec(1)= c * COS( beta )
     cVec(2)= c * ( ( COS( alpha ) - COS( beta ) * COS( gamma ) ) / SIN( gamma ) )
     Omega= a * b * c * ( SQRT( 1.d0 - COS( alpha )**2 - COS( beta )**2  &
          & - COS( gamma )**2 &
          & + 2.d0 * COS( alpha ) * COS( beta ) * COS( gamma ) ) ) 
     cVec(3)= Omega / ( a * b * SIN( gamma ) )

     WRITE(*,*) a, b, c, alpha, beta, gamma
     WRITE(*,'(3F12.6)') aVec(:)
     WRITE(*,'(3F12.6)') bVec(:)
     WRITE(*,'(3F12.6)') cVec(:)
     ! -------------------------

     WRITE(NEWCOMMENTLINE,'(A,9F12.6,2A)') 'Lattice=" ', aVec(:), bVec(:), cVec(:) &
          & , ' " Properties=species:S:1:pos:R:3 Filename= ', ciffilename
     NEWCOMMENTLINE= ADJUSTL( TRIM( NEWCOMMENTLINE ) ) 
     WRITE(*,*) 'Comment here: ------'
     WRITE(*,'(A)') NEWCOMMENTLINE

     WRITE(COMMAND,'(4A)')  "sed -i '2s/.*/", NEWCOMMENTLINE," /' ", xyzfilename
     COMMAND= ADJUSTL( TRIM( COMMAND ) ) 
     WRITE(*,*) 'Command here: ------'
     WRITE(*,'(A)')  COMMAND

     CALL SYSTEM( COMMAND )

  END DO

10 CONTINUE

  CLOSE( 100 )

END PROGRAM FINALIZE
