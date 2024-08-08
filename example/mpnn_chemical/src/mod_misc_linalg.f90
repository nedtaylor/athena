!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor and Francis Huw Davies
!!! Code part of the ARTEMIS group (Hepplestone research group).
!!! Think Hepplestone, think HRG.
!!!#############################################################################
!!! module contains various linear algebra functions and subroutines.
!!! module includes the following functions and subroutines:
!!! uvec             (unit vector of vector of any size)
!!! modu             (magnitude of vector of any size)
!!! proj             (projection operator of one vector on another)
!!! GramSchmidt      (evaluates Gram-Schmidt orthogonal vectors)
!!! cross            (cross product of two vectors)
!!! cross_matrix     (generates cross product matrix of a vector)
!!! outer_product    (performs outer_product of two vectors)
!!! vec_mat_mul      (multiply a vector with a matrix)
!!! get_vec_multiple (determines the scaling factor between two vectors)
!!!##################
!!! signed_dist      (get Signed Distance between a plane and a point)
!!! get_distance     (get the distance between two points)
!!! get_angle        (get the angle between two vectors)
!!! get_dihedral_angle (get the dihedral angle between two planes)
!!! get_area         (get the area made by two vectors)
!!! get_vol          (get the volume of a matrix)
!!! trace            (trace of a matrix of any size)
!!! det              (determinant of a 3x3 matrix)
!!! inverse          (inverse of a 3x3 matrix)
!!! rec_det          (determinant of a matrix of any size)
!!! LUdet            (determinant of a matrix of any size using LUdecomposition)
!!! LUinv            (inverse of a matrix of any size using LUdecomposition)
!!! LUdecompose      (decompose a matrix into upper and lower matrices. A=LU)
!!!##################
!!! get_spheres_overlap (get the overlap of two spheres)
!!!##################
!!! find_tf          (transformation matrix to move between two matrices)
!!! simeq            (simultaneous equation solver)
!!! IDW              (inverse distance weighting interpolate (Shepards' method))
!!! LLL_reduce       (performs LLL reduction on a basis)
!!! rotvec           (rotate vector in 3D space about x, y, z cartesian axes)
!!! rot_arb_lat      (rotate vector in 3D space about a, b, c arbitrary axes)
!!!##################
!!! gcd              (greatest common denominator (to reduce a fraction))
!!! lcm              (lowest common multiple)
!!! get_frac_denom   (convert decimal to fraction and finds lowest denominator)
!!! reduce_vec_gcd   (reduces the gcd of a vector to 1)
!!!##################
!!! gen_group        (generate group from a subset of elements)
!!!##################
!!! initialise_tetrahedra   (initialise tetrahedra and their weights)
!!!#############################################################################
module misc_linalg
  use constants_mnist, only: real32, pi
  implicit none
  integer, parameter, private :: QuadInt_K = selected_int_kind (16)

  interface get_angle
     procedure get_angle_from_points, get_angle_from_vectors
  end interface get_angle

  interface get_dihedral_angle
     procedure get_dihedral_angle_from_points, get_dihedral_angle_from_vectors
  end interface get_dihedral_angle

  interface gcd
     procedure gcd_vec,gcd_num
  end interface gcd

  interface vec_mat_mul
     procedure ivec_dmat_mul,rvec_dmat_mul
  end interface vec_mat_mul


  private

  public :: uvec, modu, proj, GramSchmidt, cross, cross_matrix, outer_product
  public :: vec_mat_mul, get_vec_multiple
  public :: signed_dist
  public :: get_distance, get_angle, get_dihedral_angle, get_area, get_vol
  public :: trace, det
  public :: inverse_2x2, inverse_3x3, inverse
  public :: rec_det, LUdet, LUinv, LUdecompose
  public :: get_spheres_overlap
  public :: find_tf, simeq, IDW, IDW_arr_fmt, IDW_grid
  public :: LLL_reduce, rotvec, rot_arb_lat
  public :: gcd, lcm, get_frac_denom, reduce_vec_gcd
  public :: gen_group
  public :: initialise_tetrahedra


!!!updated 2021/12/09


contains
!!!#####################################################
!!! finds unit vector of an arbitrary vector
!!!#####################################################
  pure function uvec(vec)
    implicit none
    real(real32),dimension(:), intent(in)::vec
    real(real32),allocatable,dimension(:)::uvec
    allocate(uvec(size(vec)))
    uvec=vec/modu(vec)
  end function uvec
!!!#####################################################


!!!#####################################################
!!! finds modulus of an arbitrary length vector
!!!#####################################################
  pure function modu(vec)
    implicit none
    real(real32),dimension(:), intent(in)::vec
    real(real32)::modu
    modu=abs(sqrt(sum(vec(:)**2)))
  end function modu
!!!#####################################################


!!!#####################################################
!!! projection operator
!!!#####################################################
!!! projection of v on u
  pure function proj(u,v)
    implicit none
    real(real32), dimension(:), intent(in) :: u,v
    real(real32), allocatable, dimension(:) :: proj

    allocate(proj(size(u,dim=1)))
    proj = u*dot_product(v,u)/dot_product(u,u)

  end function proj
!!!#####################################################


!!!#####################################################
!!! Gram-Schmidt process
!!!#####################################################
!!! assumes basis(n,m) is a basis of n vectors, each ...
!!! ... of m-dimensions
!!! rmc = row major order
  function GramSchmidt(basis,normalise,cmo) result(u)
    implicit none
    integer :: num,dim,i,j
    real(real32), allocatable, dimension(:) :: vtmp
    real(real32), dimension(:,:), intent(in) :: basis
    real(real32), allocatable, dimension(:,:) :: u
    logical, optional, intent(in) :: cmo
    logical, optional, intent(in) :: normalise


    !! sets up array dimensions of Gram-Schmidt basis
    if(present(cmo))then
       if(cmo)then
          write(0,'("Column Major Order Gram-Schmidt &
               &not yet set up")')
          write(0,'("Stopping...")')
          stop
          num = size(basis(1,:),dim=1)
          dim = size(basis(:,1),dim=1)
          allocate(u(dim,num))
          goto 10
       end if
    end if
    num = size(basis(:,1),dim=1)
    dim = size(basis(1,:),dim=1)
    allocate(u(num,dim))
    
10  allocate(vtmp(dim))

    !! Evaluates the Gram-Schmidt basis
    u(1,:) = basis(1,:)
    do i=2,num
       vtmp = 0._real32
       do j=1,i-1,1
          vtmp(:) = vtmp(:) + proj(u(j,:),basis(i,:))
       end do
       u(i,:) = basis(i,:) - vtmp(:)
    end do


    !! Normalises new basis if required
    if(present(normalise))then
       if(normalise)then
          do i=1,num
             u(i,:) = u(i,:)/modu(u(i,:))
          end do
       end if
    end if


  end function GramSchmidt
!!!#####################################################


!!!#####################################################
!!! cross product
!!!#####################################################
  pure function cross(a,b)
    implicit none
    real(real32), dimension(3) :: cross
    real(real32), dimension(3), intent(in) :: a,b

    cross(1) = a(2)*b(3) - a(3)*b(2)
    cross(2) = a(3)*b(1) - a(1)*b(3)
    cross(3) = a(1)*b(2) - a(2)*b(1)

    return
  end function cross
!!!#####################################################


!!!#####################################################
!!! cross product matrix
!!!#####################################################
!!! a = (a1,a2,a3)
!!! 
!!!         (  0  -a3  a2 )
!!! [a]_x = (  a3  0  -a1 )
!!!         ( -a2  a1  0  )
!!!#####################################################
  pure function cross_matrix(a)
    implicit none
    real(real32), dimension(3,3) :: cross_matrix
    real(real32), dimension(3), intent(in) :: a

    cross_matrix=0._real32

    cross_matrix(1,2) = -a(3)
    cross_matrix(1,3) =  a(2)
    cross_matrix(2,3) = -a(1)

    cross_matrix(2,1) =  a(3)
    cross_matrix(3,1) = -a(2)
    cross_matrix(3,2) =  a(1)

    return
  end function cross_matrix
!!!#####################################################


!!!#####################################################
!!! outer product
!!!#####################################################
  pure function outer_product(a,b)
    implicit none
    integer :: j
    real(real32), dimension(:), intent(in) :: a,b
    real(real32),allocatable,dimension(:,:)::outer_product
   
    allocate(outer_product(size(a),size(b)))

    do j=1,size(b)
       outer_product(:,j)=a(:)*b(j)
    end do

    return
  end function outer_product
!!!#####################################################


!!!#####################################################
!!! function to multiply a vector and a matrix
!!!#####################################################
  function ivec_dmat_mul(a,mat) result(vec)
    implicit none
    integer :: j
    integer, dimension(:) :: a
    real(real32), dimension(:,:) :: mat
    real(real32),allocatable,dimension(:) :: vec

    vec=0._real32
    allocate(vec(size(a)))
    do j=1,size(a)
       vec(:)=vec(:)+dble(a(j))*mat(j,:)
    end do

    return
  end function ivec_dmat_mul
!!!-----------------------------------------------------
!!!-----------------------------------------------------
  function rvec_dmat_mul(a,mat) result(vec)
    implicit none
    integer :: j
    real(real32), dimension(:) :: a
    real(real32), dimension(:,:) :: mat
    real(real32),allocatable,dimension(:) :: vec

    vec=0._real32
    allocate(vec(size(a)))
    do j=1,size(a)
       vec(:)=vec(:)+a(j)*mat(j,:)
    end do

    return
  end function rvec_dmat_mul
!!!#####################################################


!!!#####################################################
!!! get vec_multiple
!!!#####################################################
  function get_vec_multiple(a,b) result(multi)
    implicit none
    integer :: i
    real(real32) :: multi
    real(real32), dimension(:) :: a,b
    
    multi=1._real32
    do i=1,size(a)
       if(a(i).eq.0._real32.or.b(i).eq.0._real32) cycle
       multi=b(i)/a(i)
       exit
    end do

    checkloop: do i=1,size(a)
       if(a(i).eq.0._real32.or.b(i).eq.0._real32) cycle
       if(abs(a(i)*multi-b(i)).gt.1.D-8)then

          multi=0._real32
          exit checkloop
       end if
    end do checkloop

    return
  end function get_vec_multiple
!!!#####################################################


!!!#############################################################################
!!!#############################################################################
!!!  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *
!!!#############################################################################
!!!#############################################################################



!!!#####################################################
!!! returns signed distance between plane and point
!!!#####################################################
  function signed_dist(plane_vec,plane_point,point)
    implicit none
    real(real32) :: signed_dist
    real(real32), dimension(3) :: plane_vec,plane_point,point

    signed_dist = dot_product(point - plane_point,plane_vec)

    return
  end function signed_dist
!!!#####################################################


!!!#####################################################
!!! returns distance between two points
!!!#####################################################
  pure function get_distance(point1,point2) result(distance)
    implicit none
    real(real32) :: distance
    real(real32), dimension(3), intent(in) :: point1,point2

    distance = modu(point1-point2)

    return
  end function get_distance
!!!#####################################################


!!!#####################################################
!!! returns angle between two vectors
!!!#####################################################
  pure function get_angle_from_vectors(vec1,vec2) result(angle)
    implicit none
    real(real32), dimension(3), intent(in) :: vec1,vec2
    real(real32) :: angle

    angle = acos( dot_product(vec1,vec2)/&
         ( modu(vec1) * modu(vec2) ))
    if (isnan(angle)) angle = 0._real32

    return
  end function get_angle_from_vectors
!!!-----------------------------------------------------
!!! get the angle between vectors point1point2 and point2point3
!!! i.e. follow the path of point1 -> point2 -> point3
!!!-----------------------------------------------------
  pure function get_angle_from_points(point1, point2, point3) result(angle)
    implicit none
    real(real32), dimension(3), intent(in) :: point1, point2, point3
    real(real32) :: angle

    angle = acos( ( dot_product( point2 - point1, point3 - point2 ) ) / &
         ( modu( point2 - point1 ) * modu( point3 - point2 ) ) )
    if(isnan(angle)) angle = 0._real32
  end function get_angle_from_points
!!!#####################################################


!!!#####################################################
!!! returns the dihedral angle between the plane defined by the vectors ...
!!! vec1 x vec2 and the vector vec3
!!!#####################################################
  pure function get_dihedral_angle_from_vectors(vec1,vec2,vec3) result(angle)
    implicit none
    real(real32), dimension(3), intent(in) :: vec1,vec2,vec3
    real(real32) :: angle

    angle = get_angle(cross(vec1, vec2), vec3)

  end function get_dihedral_angle_from_vectors
!!!-----------------------------------------------------
!!! get the angle between the plane defined by ...
!!! ... point1point2point3 and the vector point2point4
!!!-----------------------------------------------------
  pure function get_dihedral_angle_from_points(point1, point2, point3, point4) &
         result(angle)
     implicit none
     real(real32), dimension(3), intent(in) :: point1, point2, point3, point4
     real(real32) :: angle
  
     angle = get_angle(cross(point2 - point1, point3 - point2), point4 - point2)
  
  end function get_dihedral_angle_from_points
!!!#####################################################


!!!#####################################################
!!! returns area made by two vectors
!!!#####################################################
  pure function get_area(a,b) result(area)
    implicit none
    real(real32), dimension(3), intent(in) :: a,b
    real(real32) :: area
    real(real32), dimension(3) :: vec

    vec = cross(a,b)
    area = sqrt(dot_product(vec,vec))

    return
  end function get_area
!!!#####################################################


!!!#####################################################
!!! returns volume of a lattice
!!!#####################################################
  function get_vol(lat) result(vol)
    implicit none
    integer :: n,i,j,k,l
    real(real32) :: vol,scale
    real(real32), dimension(3,3) :: lat
    real(real32), dimension(3) :: a,b,c

    a=lat(1,:)
    b=lat(2,:)
    c=lat(3,:)
    vol = 0._real32;scale = 1._real32
    i=1;j=2;k=3
1   do n=1,3
       vol = vol+scale*a(i)*b(j)*c(k)
       l=i;i=j;j=k;k=l
    end do
    i=2;j=1;k=3;scale=-scale
    if(scale<0._real32) goto 1

    return
  end function get_vol
!!!#####################################################


!!!#####################################################
!!! finds trace of an arbitrary dimension square matrix
!!!#####################################################
  function trace(mat)
    implicit none
    integer::j
    real(real32),dimension(:,:)::mat
    real(real32)::trace
    do j=1,size(mat,1)
       trace=trace+mat(j,j)
    end do
  end function trace
!!!#####################################################


!!!#####################################################
!!! returns determinant of 3 x 3 matrix
!!!#####################################################
  function det(mat)
    implicit none
    real(real32) :: det
    real(real32), dimension(3,3) :: mat

    det=mat(1,1)*mat(2,2)*mat(3,3)-mat(1,1)*mat(2,3)*mat(3,2)&
         - mat(1,2)*mat(2,1)*mat(3,3)+mat(1,2)*mat(2,3)*mat(3,1)&
         + mat(1,3)*mat(2,1)*mat(3,2)-mat(1,3)*mat(2,2)*mat(3,1)

  end function det
!!!#####################################################


!!!#####################################################
!!! returns inverse of 2x2 or 3x3 matrix
!!!#####################################################
  pure function inverse(mat)
    implicit none
    real(real32), dimension(:,:), intent(in) :: mat
    real(real32), dimension(size(mat(:,1),dim=1),size(mat(1,:),dim=1)) :: inverse

    if(size(mat(1,:),dim=1).eq.2)then
       inverse=inverse_2x2(mat)
    elseif(size(mat(1,:),dim=1).eq.3)then
       inverse=inverse_3x3(mat)
    end if

  end function inverse
!!!#####################################################


!!!#####################################################
!!! returns inverse of 2 x 2 matrix
!!!#####################################################
  pure function inverse_2x2(mat) result(inverse)
    implicit none
    real(real32) :: det
    real(real32), dimension(2,2) :: inverse
    real(real32), dimension(2,2), intent(in) :: mat

    det=mat(1,1)*mat(2,2)-mat(1,2)*mat(2,1)
    !if(det.eq.0._real32)then
    !   write(0,'("ERROR: Internal error in inverse_2x2")')
    !   write(0,'(2X,"inverse_2x2 in mod_misc_linalg found determinant of 0")')
    !   write(0,'(2X,"Exiting...")')
    !   stop
    !end if

    inverse(1,1)=+1._real32/det*(mat(2,2))
    inverse(2,1)=-1._real32/det*(mat(1,2))
    inverse(1,2)=-1._real32/det*(mat(2,1))
    inverse(2,2)=+1._real32/det*(mat(1,1))

  end function inverse_2x2
!!!#####################################################


!!!#####################################################
!!! returns inverse of 3 x 3 matrix
!!!#####################################################
  pure function inverse_3x3(mat) result(inverse)
    implicit none 
    real(real32) :: det
    real(real32), dimension(3,3) :: inverse
    real(real32), dimension(3,3), intent(in) :: mat

    det=mat(1,1)*mat(2,2)*mat(3,3)-mat(1,1)*mat(2,3)*mat(3,2)&
         - mat(1,2)*mat(2,1)*mat(3,3)+mat(1,2)*mat(2,3)*mat(3,1)&
         + mat(1,3)*mat(2,1)*mat(3,2)-mat(1,3)*mat(2,2)*mat(3,1)

    !if(det.eq.0._real32)then
    !   write(0,'("ERROR: Internal error in inverse_3x3")')
    !   write(0,'(2X,"inverse_3x3 in mod_misc_linalg found determinant of 0")')
    !   write(0,'(2X,"Exiting...")')
    !   stop
    !end if

    inverse(1,1)=+1._real32/det*(mat(2,2)*mat(3,3)-mat(2,3)*mat(3,2))
    inverse(2,1)=-1._real32/det*(mat(2,1)*mat(3,3)-mat(2,3)*mat(3,1))
    inverse(3,1)=+1._real32/det*(mat(2,1)*mat(3,2)-mat(2,2)*mat(3,1))
    inverse(1,2)=-1._real32/det*(mat(1,2)*mat(3,3)-mat(1,3)*mat(3,2))
    inverse(2,2)=+1._real32/det*(mat(1,1)*mat(3,3)-mat(1,3)*mat(3,1))
    inverse(3,2)=-1._real32/det*(mat(1,1)*mat(3,2)-mat(1,2)*mat(3,1))
    inverse(1,3)=+1._real32/det*(mat(1,2)*mat(2,3)-mat(1,3)*mat(2,2))
    inverse(2,3)=-1._real32/det*(mat(1,1)*mat(2,3)-mat(1,3)*mat(2,1))
    inverse(3,3)=+1._real32/det*(mat(1,1)*mat(2,2)-mat(1,2)*mat(2,1))

  end function inverse_3x3
!!!#####################################################


!!!#####################################################
!!! determinant function
!!!#####################################################
  recursive function rec_det(a,n) result(res)
    implicit none
    integer :: i, sign
    real(real32) :: res
    integer, intent(in) :: n
    real(real32), dimension(n,n), intent(in) :: a
    real(real32), dimension(n-1, n-1) :: tmp

    if(n.eq.1) then
       res = a(1,1)
    else
       res = 0
       sign = 1
       do i=1, n
          tmp(:,:(i-1))=a(2:,:i-1)
          tmp(:,i:)=a(2:,i+1:)
          res=res+sign*a(1,i)*rec_det(tmp,n-1)
          sign=-1._real32*sign
       end do
    end if

    return
  end function rec_det
!!!#####################################################


!!!#####################################################
!!! determinant of input matrix via LU decomposition
!!!#####################################################
!!! L = lower
!!! U = upper
!!! inmat = input nxn matrix
!!! LUdet = determinant of inmat
!!! LUdet = (-1)**N * prod(L(i,i)*U(i,i))
  function LUdet(inmat)
    implicit none
    integer :: i,N
    real(real32) :: LUdet
    real(real32), dimension(:,:) :: inmat
    real(real32), dimension(size(inmat,1),size(inmat,1)) :: L,U

    L=0._real32
    U=0._real32
    N=size(inmat,1)
    call LUdecompose(inmat,L,U)

    LUdet=(-1._real32)**N
    do i=1,N
       LUdet=LUdet*L(i,i)*U(i,i)
    end do

    return
  end function LUdet
!!!#####################################################


!!!#####################################################
!!! inverse of n x n matrix
!!!#####################################################
!!! doesn't work if a diagonal element = 0
!!! L = lower
!!! U = upper
!!! inmat = input nxn matrix
!!! LUinv = output nxn inverse of matrix
!!! Lz=b
!!! Ux=z
!!! x=column vectors of the inverse matrix
  function LUinv(inmat)
    implicit none
    integer :: i,m,N
    real(real32), dimension(:,:) :: inmat
    real(real32), dimension(size(inmat,1),size(inmat,1)) :: LUinv
    real(real32), dimension(size(inmat,1),size(inmat,1)) :: L,U
    real(real32), dimension(size(inmat,1)) :: c,z,x

    L=0._real32
    U=0._real32
    N=size(inmat,1)
    call LUdecompose(inmat,L,U)

!!! Lz=c
!!! c are column vectors of the identity matrix
!!! uses forward substitution to solve
    do m=1,N
       c=0._real32
       c(m)=1._real32

       z(1)=c(1)
       do i=2,N
          z(i)=c(i)-dot_product(L(i,1:i-1),z(1:i-1))
       end do


!!! Ux=z
!!! x are the rows of the inversion matrix
!!! uses backwards substitution to solve
       x(N)=z(N)/U(N,N)
       do i=N-1,1,-1
          x(i)=z(i)-dot_product(U(i,i+1:N),x(i+1:N))
          x(i)= x(i)/U(i,i)
       end do

       LUinv(:,m)=x(:)
    end do

    return
  end function LUinv
!!!#####################################################


!!!#####################################################
!!! A=LU matrix decomposer
!!!#####################################################
!!! Method: Based on Doolittle LU factorization for Ax=b
!!! doesn't work if a diagonal element = 0
!!! L = lower
!!! U = upper
!!! inmat = input nxn matrix
  subroutine LUdecompose(inmat,L,U)
    implicit none
    integer :: i,j,N
    real(real32), dimension(:,:) :: inmat,L,U
    real(real32), dimension(size(inmat,1),size(inmat,1)) :: mat

    N=size(inmat,1)
    mat=inmat
    L=0._real32
    U=0._real32

    do j=1,N
       L(j,j)=1._real32
    end do
!!! Solves the lower matrix
    do j=1,N-1
       do i=j+1,N
          L(i,j)=mat(i,j)/mat(j,j)
          mat(i,j+1:N)=mat(i,j+1:N)-L(i,j)*mat(j,j+1:N)
       end do
    end do

!!! Equates upper half of remaining mat to upper matrix
    do j=1,N
       do i=1,j
          U(i,j)=mat(i,j)
       end do
    end do

    return
  end subroutine LUdecompose
!!!#####################################################


!!!#############################################################################
!!!#############################################################################
!!!  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *
!!!#############################################################################
!!!#############################################################################


!!!#####################################################
!!! get overlap of two spheres
!!!#####################################################
  pure function get_spheres_overlap(radius_1,radius_2,separation) &
       result(overlap)
    implicit none
    real(real32), intent(in) :: radius_1, radius_2, separation
    real(real32) :: overlap

    real(real32) :: distance_1, distance_2, cap_volume1, cap_volume2


    overlap = 0._real32

    !! check for overlap
    if (separation .ge. radius_1 + radius_2) return
  
    !! check for completely enclosed sphere
    if ( separation + radius_1 .le. radius_2 .or. &
         separation + radius_2 .le. radius_1) then
        overlap = (4._real32/3._real32) * pi * &
             min(radius_1, radius_2) ** 3._real32
        return
    end if
  
   !! get distance from centre of sphere to plane of intersection
   distance_1 = ( radius_1 ** 2._real32 - &
                  radius_2 ** 2._real32 + &
                  separation ** 2._real32 ) / ( 2._real32 * separation )
   distance_2 = separation - distance_1
  
   !! get the volume of the spherical caps
   cap_volume1 = ( 1._real32 / 3._real32 ) * pi * &
                 distance_1 ** 2._real32 * &
                 ( 3._real32 * radius_1 - distance_1)
   cap_volume2 = ( 1._real32 / 3._real32 ) * pi * &
                 distance_2 ** 2._real32 * &
                 ( 3._real32 * radius_2 - distance_2)
  
   !! get the volume of the intersection
   overlap = cap_volume1 + cap_volume2

  end function get_spheres_overlap
!!!#####################################################


!!!#############################################################################
!!!#############################################################################
!!!  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *
!!!#############################################################################
!!!#############################################################################


!!!#####################################################
!!! find transformation matrix between two matrices
!!! A=mat1; B=mat2; T=find_tf
!!! A T = B
!!! A^-1 A T = A^-1 B
!!! T = A^-1 B
!!!#####################################################
  function find_tf(mat1,mat2) result(tf)
    implicit none
    real(real32), dimension(:,:) :: mat1,mat2
    real(real32), allocatable, dimension(:,:) :: tf

    allocate(tf(size(mat2(:,1),dim=1),size(mat1(1,:),dim=1)))
    tf=matmul(inverse(mat1),mat2)


  end function find_tf
!!!#####################################################


!!!#####################################################
!!! simultaneous equation solver for n dimensions
!!!#####################################################
!!! P     = power series equation in matrix
!!! invP  = inverse of the power series matrix
!!! qX    = the x values of the power series with a ...
!!!         ... size equal to order
!!! qY    = the Y values for the n simult eqns
!!! simeq = the coefficients of the powers of ...
!!!                ... qX with highest power simeq(1)
!!! f(qX)=qY
!!! qA P(qX) = qY (in matrix form)
!!! hence, qA=qY P^-1
  function simeq(qX,qY)
    implicit none
    integer :: i,j,n,loc
    real(real32), dimension(:) :: qX,qY
    real(real32), dimension(size(qY)) :: funcY
    real(real32), dimension(size(qY)) :: simeq,tmpqY
    real(real32), dimension(size(qY),size(qY)) :: P,invP,tmpP


    n=size(qX)
    funcy=qY
    P=0._real32
    do i=1,n
       do j=1,n
          P(i,j)=(qX(i)**dble(n-j))
       end do
    end do
    !  P(1,1)=qX(1)**2 ;P(1,2)=qX(1)   ;P(1,3)=1.0;
    !  P(2,1)=qX(2)**2 ;P(2,2)=qX(2)   ;P(2,3)=1.0;
    !  P(3,2)=qX(3)**2 ;P(3,2)=qX(3)   ;P(3,3)=1.0;

    if(any(qX.lt.1.D-5)) then
       loc=minloc(abs(qX),dim=1)
       tmpqY=funcY
       tmpP=P
       funcY(loc)=tmpqY(n)
       funcY(n)=tmpqY(loc)
       P(loc,:)=tmpP(n,:)
       P(n,:)=tmpP(loc,:)
    end if

    !  invP=inverse(P)
    invP=LUinv((P))
    !  invP=LUinv(dble(P))
    simeq=matmul(invP,funcY)

  end function simeq
!!!#####################################################


!!!#####################################################
!!! IDW interpolation method (Shepard's method)
!!!#####################################################
!!! wf  = weight function
!!! pwr = power
  function IDW(coord, dataset, valset, power) result(val)
    implicit none
    integer :: i, j, ndata
    real(real32) :: val, pwr, tot_weight
    real(real32), dimension(:) :: coord
    real(real32), dimension(:) :: valset
    real(real32), dimension(size(coord)) :: weight
    real(real32), dimension(:,:) :: dataset

    real(real32), optional, intent(in) :: power

    !! check 
    if(present(power))then
       pwr = power
    else
       pwr = 2._real32
    end if

    if(pwr.lt.size(dataset, dim=2))then
       write(0,*) "WARNING: power law for IDW set\&
            &than dataset dimension."
       call exit()
    end if

    ndata = size(dataset, dim=1)
    tot_weight = 0._real32
    do i=1,ndata
       weight(i) = modu(coord(:) - dataset(i,:))**pwr
       tot_weight = tot_weight + weight(i)
    end do
    weight = weight/tot_weight

    val = 0._real32
    do i=1,ndata
       val = val + valset(i)*weight(i)
    end do

  end function IDW
!!!-----------------------------------------------------
  function IDW_arr_fmt(coord, lat, dataset, power) result(val)
    implicit none
    integer :: i, j, k
    integer :: imax, jmax, kmax
    real(real32) :: val, pwr, tot_weight
    real(real32), dimension(3) :: tmp_coord
    real(real32), dimension(3,3) :: lat

    real(real32), dimension(:) :: coord
    real(real32), dimension(:,:,:) :: dataset
    real(real32), allocatable, dimension(:,:,:) :: weight

    real(real32), optional, intent(in) :: power

    !! check 
    if(present(power))then
       pwr = power
    else
       pwr = 2._real32
    end if
    imax = size(dataset,dim=1)
    jmax = size(dataset,dim=2)
    kmax = size(dataset,dim=3)

    if(pwr.lt.2)then
       write(0,*) "WARNING: power law for IDW lower &
            &than dataset dimension."
       write(0,*) dataset
       write(0,*) pwr
       call exit()
    end if

    allocate(weight(imax,jmax,kmax))
    tot_weight = 0._real32
    do i=1,imax
       do j=1,jmax
          do k=1,kmax
             tmp_coord = matmul(lat,real([i,j,k]))
             weight(i,j,k) = modu(coord(:) - tmp_coord)**pwr
             tot_weight = tot_weight + weight(i,j,k)
          end do
       end do
    end do
    weight = weight/tot_weight

    val = 0._real32
    do i=1,imax
       do j=1,jmax
          do k=1,kmax
             val = val + dataset(i,j,k)*weight(i,j,k)
          end do
       end do
    end do

  end function IDW_arr_fmt
!!!-----------------------------------------------------
  function IDW_grid(grid, lat, dataset, power, tol) result(dataset_new)
    implicit none
    integer :: i, j, k, in, jn, kn, n, ig
    integer :: imax, jmax, kmax, nmax
    real(real32) :: pwr, weight, tot_weight, dist, utol
    integer, dimension(3) :: ijk, ijk_max
    real(real32), dimension(3) :: tmp_coord1,tmp_coord2,diff
    real(real32), dimension(3,3) :: lat
    integer, allocatable, dimension(:,:) :: grid_list
    real(real32), allocatable, dimension(:,:) :: grid_coord
    real(real32), allocatable, dimension(:,:,:) :: dataset_new
    real(real32), allocatable, dimension(:,:,:) :: dist_arr, weight_arr, tot_weight_arr
    real(real32), allocatable, dimension(:,:,:,:) :: coord_arr1,coord_arr2

    integer, dimension(3), intent(in) :: grid
    real(real32), dimension(:,:,:), intent(in) :: dataset

    real(real32), optional, intent(in) :: power
    real(real32), optional, intent(in) :: tol


    if(present(power))then
       pwr = power
    else
       pwr = 2._real32
    end if
    if(present(tol)) then
       utol = tol
    else
       utol = 8._real32
    end if

    if(pwr.lt.2)then
       write(0,*) "WARNING: power law for IDW lower &
            &than dataset dimension."
       write(0,*) shape(dataset)
       write(0,*) pwr
       call exit()
    end if

    nmax = 1
    do i=1,3
       ijk_max(i) = size(dataset,dim=i)
       nmax = nmax * min(ijk_max(i), 2*ceiling((ijk_max(i)-1)*utol/modu(lat(i,:)))+1)
    end do
    allocate(grid_list(nmax,3))
    allocate(grid_coord(nmax,3))

    n = 0
    do i=1,ijk_max(1)
       tmp_coord1(1) = real(i-1)/ijk_max(1)
       do j=1,ijk_max(2)
          tmp_coord1(2) = real(j-1)/ijk_max(2)
          do k=1,ijk_max(3)
             !from 1, 1, 1
             tmp_coord1(3) = real(k-1)/ijk_max(3)
             diff = tmp_coord1 - ceiling(tmp_coord1-0.5_real32)
             dist = modu(matmul(diff,lat))
             if(dist.gt.utol) cycle
             
             n = n + 1
             if(n.gt.nmax)then
                write(0,*) "ERROR: Number exceeded"
                call exit()
             end if
             grid_list(n,:) = [i,j,k]
             grid_coord(n,:) = tmp_coord1
             
          end do
       end do
    end do

    allocate(dataset_new(grid(1),grid(2),grid(3)))
    dataset_new = 0._real32
    do in=1,grid(1)
       tmp_coord1(1) = real(in-1)/grid(1)
       tmp_coord2(1) = mod(tmp_coord1(1)*real(ijk_max(1)),1._real32)
       do jn=1,grid(2)
          tmp_coord1(2) = real(jn-1)/grid(2)
          tmp_coord2(2) = mod(tmp_coord1(2)*real(ijk_max(2)),1._real32)
          knloop: do kn=1,grid(3)
             tmp_coord1(3) = real(kn-1)/grid(3)
             tmp_coord2(3) = mod(tmp_coord1(3)*real(ijk_max(3)),1._real32)
             
             
             tot_weight = 0._real32
             !do i=1,imax
             !   tmp_coord2(1) = real(i-1)/imax
             !   do j=1,jmax
             !      tmp_coord2(2) = real(j-1)/jmax
             !      do k=1,kmax
             !         tmp_coord2(3) = real(k-1)/kmax
             !         diff = tmp_coord1 - tmp_coord2
             !         diff = diff - ceiling(diff-0.5_real32)
             !         dist = modu(matmul(diff,lat))
             !         if(dist.gt.utol) cycle
             do ig=1,n
                
                do i=1,3
                   ijk(i) = grid_list(ig,i) + nint(tmp_coord1(i) * ijk_max(i))
                   if(ijk(i).gt.ijk_max(i)) ijk(i) = ijk(i) - ijk_max(i)
                end do
                if(any(ijk.le.0))then
                   write(0,*) "ERROR: ijk out of limit"
                   call exit()
                end if
                !write(110,*) grid_list(ig,:), in,jn,kn, ijk

                
                diff = grid_coord(ig,:) - tmp_coord2
                diff = diff - ceiling(diff-0.5_real32)
                dist = modu(matmul(diff,lat))

                

                if(dist.eq.0._real32)then
                   dataset_new(in,jn,kn) = dataset(ijk(1),ijk(2),ijk(3))
!!! WRONG! NEED TO WORK OUT COORD RELATIVE TO THIS ONE
!!! SAME GOES FOR DIST AND WEIGHT
                   cycle knloop
                end if
                weight = 1._real32/(dist**pwr)
                tot_weight = tot_weight + weight
                dataset_new(in,jn,kn) = dataset_new(in,jn,kn) + &
                     dataset(ijk(1),ijk(2),ijk(3))*weight
             end do
             !write(110,*)
             !      end do
             !   end do
             !end do
             dataset_new(in,jn,kn) = dataset_new(in,jn,kn)/tot_weight
          end do knloop
       end do
    end do
    !dataset_new = dataset_new * (imax*jmax*kmax) / product(grid) !!! feels odd that it's this way round
    !dataset_new = dataset_new/&
    !     (sum(dataset_new)*modu(lat(1,:)/grid(1))*modu(lat(2,:)/grid(2))*modu(lat(3,:)/grid(3))) *&
    !     sum(dataset)*(modu(lat(1,:)/imax)*modu(lat(2,:)/jmax)*modu(lat(3,:)/kmax))

    

  end function IDW_grid
!!!#####################################################


!!!#####################################################
!!! Lenstra-Lenstra-Lovász reduction
!!!#####################################################
!!! LLL algorithm based on the one found on Wikipedia, ...
!!! ... which is based on Hoffstein, Pipher and Silverman 2008
!!! https://en.wikipedia.org/wiki/Lenstra–Lenstra–Lovász_lattice_basis_reduction_algorithm
  function LLL_reduce(basis,delta) result(obas)
    implicit none
    integer :: num,dim,i,j,k,loc
    real(real32) :: d,dtmp
    real(real32), allocatable, dimension(:) :: vtmp,mag_bas
    real(real32), allocatable, dimension(:,:) :: mu,GSbas,obas

    real(real32), dimension(:,:), intent(in) :: basis
    real(real32), optional, intent(in) :: delta


    !! set up the value for delta
    if(present(delta))then
       d = delta
    else
       d = 0.75D0
    end if
    
    !! allocate and initialise arrays
    num = size(basis(:,1),dim=1)
    dim = size(basis(1,:),dim=1)
    allocate(vtmp(dim))
    allocate(mag_bas(num))
    allocate(obas(num,dim))
    obas = basis

    !! reduce the gcd of the vectors
    do i=1,num
       obas(i,:) = reduce_vec_gcd(obas(i,:))
       mag_bas(i) = modu(obas(i,:))
    end do
    
    !! sort basis such that b1 is smallest
    do i=1,num-1,1
       loc = maxloc(mag_bas(i:num),dim=1) + i - 1
      if(loc.eq.i) cycle
       dtmp = mag_bas(i)
       mag_bas(i) = mag_bas(loc)
       mag_bas(loc) = dtmp
    
       vtmp = obas(i,:)
       obas(i,:) = obas(loc,:)
       obas(loc,:) = vtmp
    end do

    !! set up Gram-Schmidt process orthogonal basis
    allocate(GSbas(num,dim))
    GSbas = GramSchmidt(obas)

    !! set up the Gram-Schmidt coefficients
    allocate(mu(num,num))
    mu = get_mu(obas,GSbas)

    !! minimise the basis
    k = 2
    do while(k.le.num)

       jloop: do j=k-1,1!,-1
          if(abs(mu(k,j)).lt.0.5D0)then
             obas(k,:) = obas(k,:) - &
                  nint(mu(k,j))*obas(j,:)
             !! only need to update GSbas(k:,:) and mu
             !GSbas = GramSchmidt(obas)
             !mu = get_mu(obas,GSbas)
             call update_GS_and_mu(GSbas,mu,obas,k)
          end if
       end do jloop

       if(dot_product(GSbas(k,:),GSbas(k,:)).ge.&
            (d - mu(k,k-1)**2._real32)*&
            dot_product(GSbas(k-1,:),GSbas(k-1,:)) )then
          k = k + 1
       else
          vtmp = obas(k,:)
          obas(k,:) = obas(k-1,:)
          obas(k-1,:) = vtmp
          !GSbas = GramSchmidt(obas)
          !mu = get_mu(obas,GSbas)
          if(k.eq.1)then
             call update_GS_and_mu(GSbas,mu,obas,k)
          else
             call update_GS_and_mu(GSbas,mu,obas,k-1)
          end if
          k = max(k-1,2)
       end if

    end do


!!! Separate functions for this to run efficiently
  contains
    !!function to get the mu values
    function get_mu(bas1,bas2) result(mu)
      implicit none
      integer :: num1,num2
      real(real32), allocatable, dimension(:,:) :: mu,bas1,bas2
      num1 = size(bas1(:,1),dim=1)
      num2 = size(bas2(:,1),dim=1)

      allocate(mu(num1,num2))
      do i=1,num1
         do j=1,num2

            mu(i,j) = dot_product(bas1(i,:),bas2(j,:))/&
                 dot_product(bas2(j,:),bas2(j,:))

         end do
      end do

    end function get_mu


    !!subroutine to update Gram-Schmidt vectors and mu values
    subroutine update_GS_and_mu(GSbas,mu,basis,k)
      implicit none
      integer :: num,dim,i,j
      real(real32), allocatable, dimension(:) :: vtmp

      integer, intent(in) :: k
      real(real32), allocatable, dimension(:,:) :: GSbas,basis,mu

      num = size(basis(:,1),dim=1)
      dim = size(basis(1,:),dim=1)

      allocate(vtmp(dim))
      
      !!update Gram-Schmidt vectors
      do i=k,num,1
         vtmp = 0._real32
         do j=1,i-1,1
            vtmp(:) = vtmp(:) + proj(GSbas(j,:),basis(i,:))
         end do
         GSbas(i,:) = basis(i,:) - vtmp(:)
      end do


      !!update mu values
      mu_loop1: do i=1,num,1
         mu_loop2: do j=1,num,1
      
            if(i.lt.k.and.j.lt.k) cycle mu_loop2
            
            mu(i,j) = dot_product(basis(i,:),GSbas(j,:))/&
                 dot_product(GSbas(j,:),GSbas(j,:))
            
      
         end do mu_loop2
      end do mu_loop1

    end subroutine update_GS_and_mu


  end function LLL_reduce
!!!#####################################################


!!!#####################################################
!!! vector rotation
!!!#####################################################
  function rotvec(a,theta,phi,psi,new_length)
    implicit none
    real(real32) :: magold,theta,phi,psi
    real(real32), dimension(3) :: a,rotvec
    real(real32), dimension(3,3) :: rotmat,rotmatx,rotmaty,rotmatz
    real(real32), optional :: new_length

    !  if(phi.ne.0._real32) phi=-phi

    rotmatx=reshape((/&
         1._real32, 0._real32,   0._real32,  &
         0._real32, cos(theta), -sin(theta),&
         0._real32, sin(theta),  cos(theta)/), shape(rotmatx))
    rotmaty=reshape((/&
         cos(phi),  0._real32, sin(phi),&
         0._real32, 1._real32, 0._real32,    &
         -sin(phi), 0._real32, cos(phi)/), shape(rotmaty))
    rotmatz=reshape((/&
         cos(psi), -sin(psi),  0._real32,&
         sin(psi),  cos(psi),  0._real32,    &
         0._real32, 0._real32, 1._real32/), shape(rotmatz))


    rotmat=matmul(rotmaty,rotmatx)
    rotmat=matmul(rotmatz,rotmat)
    rotvec=matmul(a,transpose(rotmat))

    if(present(new_length))then
       magold=sqrt(dot_product(a,a))
       rotvec=rotvec*new_length/magold
    end if

    return
  end function rotvec
!!!#####################################################


!!!#####################################################
!!! vector rotation
!!!#####################################################
  function rot_arb_lat(a,lat,ang) result(vec)
    implicit none
    integer :: i
    real(real32), dimension(3) :: a,u,ang,vec
    real(real32), dimension(3,3) :: rotmat,ident,lat


    ident=0._real32
    do i=1,3
       ident(i,i)=1._real32
    end do
   
    vec=a
    do i=1,3
       u=uvec(lat(i,:))
       rotmat=&
            (cos(ang(i))*ident)+&
            (sin(ang(i)))*cross_matrix(u)+&
            (1-cos(ang(i)))*outer_product(u,u)
       vec=matmul(vec,rotmat)
    end do


    return
  end function rot_arb_lat
!!!#####################################################


!!!#############################################################################
!!!#############################################################################
!!!  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *
!!!#############################################################################
!!!#############################################################################


!!!#####################################################
!!! finds the greatest common denominator
!!!#####################################################
  function gcd_num(numer,denom) result(gcd)
    implicit none
    integer :: numer,denom
    integer :: a,b,c,gcd

    a=abs(numer)
    b=abs(denom)
    if(a.gt.b)then
       c=a
       a=b
       b=c
    end if

    if(a.eq.0)then
       gcd=b
       return
    end if

    do 
       c=mod(b,a)
       if(c.eq.0) exit
       b=a
       a=c
    end do
    gcd=a

    return
  end function gcd_num
!!!-----------------------------------------------------
!!!-----------------------------------------------------
  function gcd_vec(vec) result(gcd)
    implicit none
    integer :: i,a,b,c,dim,itmp1,loc
    integer :: gcd
    integer, dimension(:),intent(in) :: vec
    integer, allocatable, dimension(:) :: in_vec


    dim=size(vec,dim=1)
    allocate(in_vec(dim))
    in_vec=abs(vec)
    do i=1,dim
       loc=maxloc(in_vec(i:dim),dim=1)+i-1
       itmp1=in_vec(i)
       in_vec(i)=in_vec(loc)
       in_vec(loc)=itmp1
    end do

    a=in_vec(2)
    do i=1,dim
       if(in_vec(i).eq.0) exit
       b=in_vec(i)
       do 
          c=mod(b,a)
          if(c.eq.0) exit
          b=a
          a=c
       end do
    end do
    gcd=a

    return
  end function gcd_vec
!!!#####################################################


!!!#####################################################
!!! finds the lowest common multiple
!!!#####################################################
  function lcm(a,b)
    implicit none
    integer :: a,b,lcm

    lcm=abs(a*b)/gcd(a,b)

    return
  end function lcm
!!!#####################################################


!!!#####################################################
!!! converts decimal into a fraction and finds the ...
!!! ... lowest denominator for it.
!!!#####################################################
  integer function get_frac_denom(val)
    implicit none
    integer :: i
    real(real32) :: val
    real(real32) :: a,b,c,tiny

    a=mod(val,1._real32)
    b=1._real32
    tiny=1.D-6
    i=0
    do 
       i=i+1
       if(abs(nint(1._real32/a)-(1._real32/a)).lt.tiny.and.&
            abs(nint(val*1._real32/a)-val*(1._real32/a)).lt.tiny) exit
       c=abs(b-a)
       b=a
       a=c
       if(i.ge.1000)then
          get_frac_denom=0
          return
       end if
    end do

    get_frac_denom=nint(1._real32/a)

    return
  end function get_frac_denom
!!!#####################################################


!!!#####################################################
!!! reduces the gcd of a vector to 1
!!!#####################################################
  function reduce_vec_gcd(invec) result(vec)
    implicit none
    integer :: i,a
    real(real32) :: div,old_div,tol
    real(real32), allocatable, dimension(:) :: vec,tvec
    real(real32), dimension(:), intent(in) :: invec


!!! MAKE IT DO SOMETHING IF IT CANNOT FULLY INTEGERISE

    tol=1.D-5
    allocate(vec(size(invec)))
    vec=invec
    if(any(abs(vec(:)-nint(vec(:))).gt.tol))then
       div=abs(vec(1))
       do i=2,size(vec),1
          old_div=div
          if(min(abs(vec(i)),div).lt.tol)then
             div=max(abs(vec(i)),div)
             cycle
          end if
          div=abs(modulo(max(abs(vec(i)),div),min(abs(vec(i)),div)))
          if(abs(div).lt.tol) div=min(abs(vec(i)),old_div)
       end do
    else
       a=vec(1)
       do i=2,size(vec)
          if(a.eq.0.and.int(vec(i)).eq.0) cycle
          a=gcd(a,int(vec(i)))
          if(abs(a).le.1)then
             a=1
             exit
          end if
       end do
       div=a
    end if

    if(div.eq.0._real32) return
    tvec=vec/div
    if(any(abs(tvec(:)-nint(tvec(:))).gt.tol)) return
    vec=tvec


  end function reduce_vec_gcd
!!!#####################################################


!!!#####################################################
!!! generate entire group from supplied elements
!!!#####################################################
  function gen_group(elem,mask,tol) result(group)
    implicit none
    integer :: i,j,k,nelem,ntot_elem,dim1,dim2,iter
    real(real32) :: tiny
    real(real32), allocatable, dimension(:,:) :: tmp_elem,cur_elem,apply_elem
    real(real32), allocatable, dimension(:,:,:) :: tmp_group

    real(real32), dimension(:,:,:), intent(in) :: elem
    logical, dimension(:,:), optional, intent(in) :: mask
    real(real32), allocatable, dimension(:,:,:) :: group
    real(real32), optional, intent(in) :: tol


    if(present(tol))then
       tiny = tol
    else
       tiny = 1.D-5
    end if
    nelem = size(elem,dim=1)
    dim1 = size(elem,dim=2)
    dim2 = size(elem,dim=3)
    !!! HARDCODED LIMIT OF A GROUP SIZE TO 10,000
    allocate(tmp_group(10000,dim1,dim2))
    allocate(tmp_elem(dim1,dim2))
    allocate(cur_elem(dim1,dim2))
    allocate(apply_elem(dim1,dim2))

    elem_check: do i=1,nelem
       if(abs(abs(det(elem(i,:3,:3)))-1._real32).gt.tiny)then
          write(0,'("ERROR: abs(determinant) of a supplied element is greater than one")')
          write(0,*) "Determinant:", det(elem(i,:3,:3))
          write(0,*) "Element:",i
          write(0,'(4(2X,F9.6))') elem(i,:,:)
          stop
       end if
    end do elem_check

    ntot_elem = 0
    elem_loop1: do i=1,nelem
       cur_elem(:,:) = elem(i,:,:)
       !write(0,*) "##########"
       !write(0,*)
       !write(0,*) i
       !write(0,'(4(2X,F9.6))') cur_elem(:,:)
       !write(0,*)
       if(present(mask))then
          where(mask.and.(cur_elem(:,:).lt.-tiny.or.cur_elem(:,:).ge.1._real32-tiny))
             cur_elem(:,:) = cur_elem(:,:) - floor(cur_elem(:,:)+tiny)
          end where
       elseif(dim1.eq.4)then
          where(cur_elem(4,:3).lt.-tiny.or.cur_elem(4,:3).gt.1._real32-tiny)
             cur_elem(4,:3) = cur_elem(4,:3) - floor(cur_elem(4,:3)+tiny)
          end where
       end if
       do k=1,ntot_elem
          if(all(abs(tmp_group(k,:,:)-cur_elem(:,:)).lt.tiny)) cycle elem_loop1
       end do
       ntot_elem = ntot_elem + 1
       tmp_group(ntot_elem,:,:) = cur_elem(:,:)

       elem_loop2: do j=1,nelem
          tmp_elem(:,:) = cur_elem(:,:)
          apply_elem(:,:) = elem(j,:,:)
          iter = 0
          !write(0,*) "-----------"
          !write(0,*) "new start",j
          !write(0,'(4(2X,F9.6))') tmp_elem(:,:)
          !write(0,*)
          !write(0,'(4(2X,F9.6))') apply_elem(:,:)
          !write(0,*) "-----------"
          recursive_loop: do
             iter = iter + 1
             if(iter.ge.10)then
                write(0,'("ERROR: unending loop in mod_misc_linalg.f90")')
                write(0,'(2X,"subroutine gen_group in mod_misc_linalg.f90 encountered an unending loop")')
                write(0,'(2X,"Exiting...")')
                stop
             end if
             !write(0,*) "apply element"
             !write(0,'(4(2X,F9.6))') tmp_elem(:,:)
             tmp_elem(:,:) = matmul((apply_elem(:,:)),tmp_elem(:,:))
             !write(0,*) "made element"
             !write(0,'(4(2X,F9.6))') tmp_elem(:,:)
             !write(0,*) 
             if(present(mask))then
                where(mask.and.(tmp_elem(:,:).lt.-tiny.or.tmp_elem(:,:).ge.1._real32-tiny))
                   tmp_elem(:,:) = tmp_elem(:,:) - floor(tmp_elem(:,:)+tiny)
                end where
             elseif(dim1.eq.4)then
                where(tmp_elem(4,:3).lt.-tiny.or.tmp_elem(4,:3).ge.1._real32-tiny)
                   tmp_elem(4,:3) = tmp_elem(4,:3) - floor(tmp_elem(4,:3)+tiny)
                end where
             end if
             if(abs(abs(det(tmp_elem(:3,:3)))-1._real32).gt.tiny)then
                write(0,'("ERROR: abs(determinant) of element greater than one")')
                write(0,*) "determinant:", det(tmp_elem(:3,:3))
                write(0,*) "Element:",i
                write(0,'(4(2X,F9.6))') cur_elem(:,:)
                write(0,*) "j element", j
                write(0,'(4(2X,F9.6))') apply_elem(:,:)
                write(0,*) "iteration", iter
                write(0,'(4(2X,F9.6))') tmp_elem(:,:)
                stop
             end if
             where(abs(tmp_elem).lt.tiny)
                tmp_elem = 0._real32
             end where
             if(all(abs(cur_elem(:,:)-tmp_elem(:,:)).lt.tiny)) exit recursive_loop
             do k=1,ntot_elem
                if(all(abs(tmp_group(k,:,:)-tmp_elem(:,:)).lt.tiny)) cycle recursive_loop
             end do
             ntot_elem = ntot_elem + 1
             tmp_group(ntot_elem,:,:) = tmp_elem(:,:)
          end do recursive_loop
       end do elem_loop2
          
       
    end do elem_loop1

    allocate(group(ntot_elem,dim1,dim2))
    group(:,:,:) = tmp_group(:ntot_elem,:,:)
    return




  end function gen_group
!!!#####################################################


!!!#############################################################################
!!!#############################################################################
!!!  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *
!!!#############################################################################
!!!#############################################################################


!!!#####################################################
!!! initialise tetrahedra and their weights
!!!#####################################################
 subroutine initialise_tetrahedra(tet, weight, reclat)
   implicit none
   integer :: i,t,w,w2
   integer :: ishort
   real(real32) :: rshort,rtmp1
   integer, dimension(4) :: ilist
   real(real32), dimension(4,3) :: diagonals

   real(real32), dimension(3,3), intent(in) :: reclat
   real(real32), dimension(6,4,3), intent(out) :: weight
   integer, dimension(6,4,3), intent(out) :: tet
   
   !       7-----------8
   !      /|          /|
   !     / |         / |
   !    /  |        /  |
   !   5-----------6   |
   !   |   |       |   |
   !   |   3-------|---4
   !   |  /        |  /
   !   | /         | /
   !   |/          |/
   !   1-----------2

   ! 1st get the shortest diagonal (called the main diagonal?)
   diagonals(1,:) = [ 1._real32,  1._real32,  1._real32 ] ! 1-8
   diagonals(2,:) = [-1._real32,  1._real32,  1._real32 ] ! 2-7
   diagonals(3,:) = [ 1._real32, -1._real32,  1._real32 ] ! 3-6
   diagonals(4,:) = [ 1._real32,  1._real32, -1._real32 ] ! 4-5

   rshort = modu(matmul(diagonals(1,:),reclat))
   ishort = 1
   do i=2,4,1
      rtmp1 = modu(matmul(diagonals(i,:),reclat))
      if(rtmp1.lt.rshort)then
         rshort = rtmp1
         ishort = i
      end if
   end do

   ! the shortest diagonal means those two associated points are in every single
   ! tetrahedron in a box.
   if (ishort.eq.1)then ! 1 - 8
      tet(1,:,:) = transpose(reshape( [0,0,0, 1,0,0, 1,1,0, 1,1,1], (/3,4/) )) ! 1, 2, 4, 8
      tet(2,:,:) = transpose(reshape( [0,0,0, 1,0,0, 1,0,1, 1,1,1], (/3,4/) )) ! 1, 2, 6, 8
      tet(3,:,:) = transpose(reshape( [0,0,0, 0,1,0, 1,1,0, 1,1,1], (/3,4/) )) ! 1, 3, 4, 8
      tet(4,:,:) = transpose(reshape( [0,0,0, 0,1,0, 0,1,1, 1,1,1], (/3,4/) )) ! 1, 3, 7, 8
      tet(5,:,:) = transpose(reshape( [0,0,0, 0,0,1, 1,0,1, 1,1,1], (/3,4/) )) ! 1, 5, 6, 8
      tet(6,:,:) = transpose(reshape( [0,0,0, 0,0,1, 0,1,1, 1,1,1], (/3,4/) )) ! 1, 5, 7, 8
   elseif(ishort.eq.2)then ! 2 - 7
      tet(1,:,:) = transpose(reshape( [1,0,0, 0,0,0, 0,1,0, 0,1,1], (/3,4/) )) ! 2, 1, 3, 7
      tet(2,:,:) = transpose(reshape( [1,0,0, 0,0,0, 0,0,1, 0,1,1], (/3,4/) )) ! 2, 1, 5, 7
      tet(3,:,:) = transpose(reshape( [1,0,0, 1,1,0, 0,1,0, 0,1,1], (/3,4/) )) ! 2, 4, 3, 7
      tet(4,:,:) = transpose(reshape( [1,0,0, 1,1,0, 1,1,1, 0,1,1], (/3,4/) )) ! 2, 4, 8, 7
      tet(5,:,:) = transpose(reshape( [1,0,0, 1,0,1, 0,0,1, 0,1,1], (/3,4/) )) ! 2, 6, 5, 7
      tet(6,:,:) = transpose(reshape( [1,0,0, 1,0,1, 1,1,1, 0,1,1], (/3,4/) )) ! 2, 6, 8, 7
   elseif(ishort.eq.3)then ! 3 - 6
      tet(1,:,:) = transpose(reshape( [0,1,0, 1,1,0, 1,0,0, 1,0,1], (/3,4/) )) ! 3, 4, 2, 6
      tet(2,:,:) = transpose(reshape( [0,1,0, 1,1,0, 1,1,1, 1,0,1], (/3,4/) )) ! 3, 4, 8, 6
      tet(3,:,:) = transpose(reshape( [0,1,0, 0,0,0, 1,0,0, 1,0,1], (/3,4/) )) ! 3, 1, 2, 6
      tet(4,:,:) = transpose(reshape( [0,1,0, 0,0,0, 0,0,1, 1,0,1], (/3,4/) )) ! 3, 1, 5, 6
      tet(5,:,:) = transpose(reshape( [0,1,0, 0,1,1, 1,1,1, 1,0,1], (/3,4/) )) ! 3, 7, 8, 6
      tet(6,:,:) = transpose(reshape( [0,1,0, 0,1,1, 0,0,1, 1,0,1], (/3,4/) )) ! 3, 7, 5, 6
   elseif(ishort.eq.4)then ! 4 - 5
      tet(1,:,:) = transpose(reshape( [1,1,0, 0,1,0, 0,0,0, 0,0,1], (/3,4/) )) ! 4, 3, 1, 5
      tet(2,:,:) = transpose(reshape( [1,1,0, 0,1,0, 0,1,1, 0,0,1], (/3,4/) )) ! 4, 3, 7, 5
      tet(3,:,:) = transpose(reshape( [1,1,0, 1,0,0, 0,0,0, 0,0,1], (/3,4/) )) ! 4, 2, 1, 5
      tet(4,:,:) = transpose(reshape( [1,1,0, 1,0,0, 1,0,1, 0,0,1], (/3,4/) )) ! 4, 2, 6, 5
      tet(5,:,:) = transpose(reshape( [1,1,0, 1,1,1, 0,1,1, 0,0,1], (/3,4/) )) ! 4, 8, 7, 5
      tet(6,:,:) = transpose(reshape( [1,1,0, 1,1,1, 1,0,1, 0,0,1], (/3,4/) )) ! 4, 8, 6, 5
   end if

   ! set up the weighting for each tetrahedron corner
   ilist = [1,2,3,4]
   do t=1,6
      do w=1,4
         ilist = cshift(ilist,1)
         weight(t,w,:) = 0._real32
         do w2=1,3
            if(modu(real(tet(t,ilist(w2),:)-tet(t,ilist(4),:),real32)).eq.1._real32)then
               weight(t,w,:) = weight(t,w,:) + (&
                    tet(t,ilist(w2),:)-tet(t,ilist(4),:)&
                    )
            end if
         end do
      end do
   end do

   return
 end subroutine initialise_tetrahedra
!!!#####################################################

end module misc_linalg
