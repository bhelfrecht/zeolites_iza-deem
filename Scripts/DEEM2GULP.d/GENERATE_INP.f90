! Giovanni Pireddu, 2019
! Generate the inputs with the fractional coordinates of the full 
! cell with space group P 1

MODULE TOOLS

CONTAINS

  FUNCTION matinv3(A) result(B)
    !! Performs a direct calculation of the inverse of a 3×3 matrix.
    REAL(KIND= 8), INTENT(in) :: A(3,3)   !! Matrix
    REAL(KIND= 8) :: B(3,3)   !! Inverse matrix
    REAL(KIND= 8) :: detinv

    ! Calculate the inverse determinant of the matrix
    detinv = 1/(A(1,1)*A(2,2)*A(3,3) - A(1,1)*A(2,3)*A(3,2)&
         - A(1,2)*A(2,1)*A(3,3) + A(1,2)*A(2,3)*A(3,1)&
         + A(1,3)*A(2,1)*A(3,2) - A(1,3)*A(2,2)*A(3,1))

    ! Calculate the inverse of the matrix
    B(1,1) = +detinv * (A(2,2)*A(3,3) - A(2,3)*A(3,2))
    B(2,1) = -detinv * (A(2,1)*A(3,3) - A(2,3)*A(3,1))
    B(3,1) = +detinv * (A(2,1)*A(3,2) - A(2,2)*A(3,1))
    B(1,2) = -detinv * (A(1,2)*A(3,3) - A(1,3)*A(3,2))
    B(2,2) = +detinv * (A(1,1)*A(3,3) - A(1,3)*A(3,1))
    B(3,2) = -detinv * (A(1,1)*A(3,2) - A(1,2)*A(3,1))
    B(1,3) = +detinv * (A(1,2)*A(2,3) - A(1,3)*A(2,2))
    B(2,3) = -detinv * (A(1,1)*A(2,3) - A(1,3)*A(2,1))
    B(3,3) = +detinv * (A(1,1)*A(2,2) - A(1,2)*A(2,1))
  END FUNCTION matinv3

END MODULE TOOLS

PROGRAM GEN
  USE TOOLS
  IMPLICIT NONE
  REAL(KIND= 8) :: a, b, c, alpha, beta, gamma, Omega, Norm
  REAL(KIND= 8), DIMENSION(:) :: aVec(1:3), bVec(1:3), cVec(1:3), iVec(1:3)
  REAL(KIND= 8), DIMENSION(:,:) :: RotA(1:3,1:3), Amat(1:3,1:3)
  REAL(KIND= 8), DIMENSION(:,:), ALLOCATABLE :: Conf, FracConf
  CHARACTER(LEN= 2), DIMENSION(:), ALLOCATABLE :: Labels
  INTEGER :: iStruc, nAT, iAT, idir
  CHARACTER(LEN=500) :: COMMENTLINE
  CHARACTER(LEN=50) :: Strucname, inpname, xyzfilename, ciffilename, newciffilename, AW, W
  CHARACTER(LEN= 7) :: fwname

  OPEN( UNIT= 100, FILE= 'Names_xyz.txt', ACTION= 'READ' )
  OPEN( UNIT= 101, FILE= 'Names_cif.txt', ACTION= 'READ' ) 

  iStruc= 0
  DO 
     READ(100,'(A)',END=10) xyzfilename
     READ(101,'(A)',END=10) ciffilename
     ciffilename= ADJUSTL( TRIM( ciffilename ) )
     fwname= ciffilename( 1 : 7 ) 
     inpname= TRIM( fwname ) // TRIM( '.inp' )
     WRITE(*,*) inpname

     iStruc= iStruc + 1
     OPEN( UNIT= 200, FILE= inpname, ACTION= 'WRITE' )
     OPEN( UNIT= 201, FILE= xyzfilename, ACTION= 'READ' )
     OPEN( UNIT= 202, FILE= ciffilename, ACTION= 'READ' )

     WRITE(200,'(A)') 'opti conp shell'
     READ(201,*) nAT
     READ(201,*)

     DO 
        READ(202,*,END=1) AW

        SELECT CASE( AW )

        CASE( '_cell_length_a')
           BACKSPACE( 202 ) 
           READ(202,*) W, a

        CASE( '_cell_length_b')
           BACKSPACE( 202 ) 
           READ(202,*) W, b

        CASE( '_cell_length_c')
           BACKSPACE( 202 ) 
           READ(202,*) W, c

        CASE( '_cell_angle_alpha')
           BACKSPACE( 202 ) 
           READ(202,*) W, alpha

        CASE( '_cell_angle_beta')
           BACKSPACE( 202 ) 
           READ(202,*) W, beta

        CASE( '_cell_angle_gamma')
           BACKSPACE( 202 ) 
           READ(202,*) W, gamma

        END SELECT

     END DO

1    CONTINUE

     !Generate the lattice vectors
     alpha= alpha * (3.1415926535897/180.d0)
     beta= beta * (3.1415926535897/180.d0)
     gamma= gamma * (3.1415926535897/180.d0)

     aVec(1)= a
     aVec(2)= 0.d0
     aVec(3)= 0.d0

     bVec(1)= b * COS( gamma )
     bVec(2)= b * SIN( gamma )
     bVec(3)= 0.d0

     cVec(1)= c * COS( beta )
     cVec(2)= c * ( ( COS( alpha ) - COS( beta ) * COS( gamma ) ) / SIN( gamma ) )
     Omega= a * b * c * ( SQRT( 1.d0 - (COS( alpha )**2) - (COS( beta )**2)  &
          & - (COS( gamma )**2) &
          & + 2.d0 * COS( alpha ) * COS( beta ) * COS( gamma ) ) ) 
     cVec(3)= Omega / ( a * b * SIN( gamma ) )

     !WRITE(*,*) aVec(:)
     !WRITE(*,*) bVec(:)
     !WRITE(*,*) cVec(:)

     !One step convert
     RotA(:,:)= 0.d0

     RotA = TRANSPOSE( RESHAPE( (/ ( 1.d0 / a ) &
          & , (- COS( gamma ) / ( a * SIN( gamma ) ) ) &
          & , ( b * c * ( COS( alpha ) * COS( gamma  ) - COS( beta ) ) &
          & / ( Omega * SIN( gamma ) ) ) &
          & , 0.d0 &
          & , 1.d0 / ( b * SIN( gamma ) ) &
          & ,  ( a * c * ( COS( beta ) * COS( gamma  ) - COS( alpha ) ) &
          & / ( Omega * SIN( gamma ) ) ) &
          & , 0.d0 &
          & , 0.d0 &
          & , ( a * b * SIN( gamma ) / Omega ) /),                            &
          (/ SIZE(RotA, 2), SIZE(RotA, 1) /)))

     Amat = ( RESHAPE( (/ &
          & aVec(1), &
          & aVec(2), &
          & aVec(3), &
          & bVec(1), &
          & bVec(2), &
          & bVec(3), &
          & cVec(1), &
          & cVec(2), &
          & cVec(3) &
          & /), &
          & (/ SIZE(RotA, 2), SIZE(RotA, 1) /)))

     !Loop on coords
     ALLOCATE( Labels(1:nAT) )
     ALLOCATE( Conf(1:nAT,1:3) )
     ALLOCATE( FracConf(1:nAT,1:3) )
     DO iAT= 1, nAT
        READ(201,*) Labels(iAT), Conf(iAT,:)
        iVec(:)= Conf(iAT,:)        
        FracConf(iAT,:)=  MATMUL( matinv3( Amat ), iVec )
        Norm= 0.d0
        DO idir= 1, 3
           Norm= Norm + FracConf(iAT,idir)**2
           ! IF( ( ABS( FracConf(iAT,idir) ) .GT. 1.d0 ) ) THEN
           ! 
           !    WRITE(*,*) 'Fractional coordinates out of boundaries (components)'
           !    WRITE(*,*) 'Original coordinates: ', Conf(iAT,:)
           !    WRITE(*,*) 'Fractional coordinates: ', FracConf(iAT,:)
           !    !STOP
           ! END IF
        END DO
        Norm= SQRT( Norm )

     END DO

     ! Generate the input for GULP
     WRITE(200,'(A)') 'title'
     WRITE(200,'(A,I3)') 'Input based on the structure ', iStruc
     WRITE(200,'(A)') 'end'
     WRITE(200,'(A)') 'vector'
     WRITE(200,'(3F12.6)') aVec(:)
     WRITE(200,'(3F12.6)') bVec(:)
     WRITE(200,'(3F12.6)') cVec(:)
     WRITE(200,'(A)') 'cartesian'

     DO iAT= 1, nAT
        WRITE(200,'(2(A,X),3F12.7)') Labels(iAT), 'core', Conf(iAT,:)

        IF( Labels(iAT) == 'O' ) THEN
           WRITE(200,'(2(A,X),3F12.7)') Labels(iAT), 'shel', Conf(iAT,:)
        END IF
     END DO

     WRITE(200,'(A)') 'species'
     WRITE(200,'(A)') 'Si core Si' 
     WRITE(200,'(A)') 'O core O_O2-' 
     WRITE(200,'(A)') 'O shel O_O2-'
     WRITE(200,'(A)') 'library ../catlow_mod.lib'
     WRITE(Strucname,'(A,A)') 'OPT_', fwname
     WRITE(200,'(2A)') 'output xyz ' , Strucname
     WRITE(200,'(2A)') 'output cif ' , Strucname

     DEALLOCATE( Labels )
     DEALLOCATE( FracConf )
     DEALLOCATE( Conf )

     CLOSE( 200 )

  END DO

10 CONTINUE

  CLOSE( 100 )

END PROGRAM GEN
