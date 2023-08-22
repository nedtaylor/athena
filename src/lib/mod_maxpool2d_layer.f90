!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module maxpool2d_layer
  use constants, only: real12
  use base_layer, only: base_layer_type
  implicit none
  
  
  type, extends(base_layer_type) :: maxpool2d_layer_type
     !! strd = stride (step)
     !! pool = pool
     integer, dimension(2) :: pool, strd
     integer :: num_channels
     real(real12), allocatable, dimension(:,:,:) :: output
     real(real12), allocatable, dimension(:,:,:) :: di ! gradient of input (i.e. delta)
   contains
     procedure, pass(this) :: print => print_maxpool2d
     procedure, pass(this) :: forward  => forward_rank
     procedure, pass(this) :: backward => backward_rank
     procedure, private, pass(this) :: forward_3d
     procedure, private, pass(this) :: backward_3d
  end type maxpool2d_layer_type

  
  interface maxpool2d_layer_type
     pure module function layer_setup( &
          input_shape, &
          pool_size, stride) result(layer)
       integer, dimension(:), intent(in) :: input_shape
       integer, dimension(..), optional, intent(in) :: pool_size
       integer, dimension(..), optional, intent(in) :: stride
       type(maxpool2d_layer_type) :: layer
     end function layer_setup
  end interface maxpool2d_layer_type


  private
  public :: maxpool2d_layer_type


contains

!!!#############################################################################
!!! forward propagation assumed rank handler
!!!#############################################################################
  pure subroutine forward_rank(this, input)
    implicit none
    class(maxpool2d_layer_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: input

    select rank(input); rank(3)
       call forward_3d(this, input)
    end select
  end subroutine forward_rank
!!!#############################################################################


!!!#############################################################################
!!! backward propagation assumed rank handler
!!!#############################################################################
  pure subroutine backward_rank(this, input, gradient)
    implicit none
    class(maxpool2d_layer_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: input
    real(real12), dimension(..), intent(in) :: gradient

    select rank(input); rank(3)
    select rank(gradient); rank(3)
      call backward_3d(this, input, gradient)
    end select
    end select
  end subroutine backward_rank
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! set up and initialise network layer
!!!#############################################################################
  pure module function layer_setup( &
       input_shape, &
       pool_size, stride) result(layer)
    implicit none
    integer, dimension(:), intent(in) :: input_shape
    integer, dimension(..), optional, intent(in) :: pool_size
    integer, dimension(..), optional, intent(in) :: stride
    
    type(maxpool2d_layer_type) :: layer

    integer, dimension(2) :: end_idx

    
    !!-----------------------------------------------------------------------
    !! set up pool size
    !!-----------------------------------------------------------------------
    if(present(pool_size))then
       select rank(pool_size)
       rank(0)
          layer%pool = pool_size
       rank(1)
          layer%pool(1) = pool_size(1)
          if(size(pool_size,dim=1).eq.1)then
             layer%pool(2) = pool_size(1)
          elseif(size(pool_size,dim=1).eq.2)then
             layer%pool(2) = pool_size(2)
          end if
       end select
    else
       layer%pool = 2
    end if


    !!-----------------------------------------------------------------------
    !! set up stride
    !!-----------------------------------------------------------------------
    if(present(stride))then
       select rank(stride)
       rank(0)
          layer%strd = stride
       rank(1)
          layer%strd(1) = stride(1)
          if(size(stride,dim=1).eq.1)then
             layer%strd(2) = stride(1)
          elseif(size(stride,dim=1).eq.2)then
             layer%strd(2) = stride(2)
          end if
       end select
    else
       layer%strd = 2
    end if


    !!-----------------------------------------------------------------------
    !! set up number of channels, width, height
    !!-----------------------------------------------------------------------
    layer%num_channels = input_shape(3)
    layer%input_shape  = input_shape(:3)
    allocate(layer%output_shape(3))
    layer%output_shape(:2) = &
         floor( (input_shape(:2)-layer%pool)/real(layer%strd)) + 1
    layer%output_shape(3) = input_shape(3)
    

    !!-----------------------------------------------------------------------
    !! allocate output and gradients
    !!-----------------------------------------------------------------------
    allocate(layer%output(&
         layer%output_shape(1),&
         layer%output_shape(2),layer%num_channels), &
         source=0._real12)
    allocate(layer%di(&
         input_shape(1),&
         input_shape(2), input_shape(3)), &
         source=0._real12)

  end function layer_setup
!!!#############################################################################


!!!#############################################################################
!!! print layer to file
!!!#############################################################################
  subroutine print_maxpool2d(this, file)
    implicit none
    class(maxpool2d_layer_type), intent(in) :: this
    character(*), intent(in) :: file

    integer :: unit

     integer, dimension(3) :: pool, strd
     integer :: num_channels

    !! open file with new unit
    !!--------------------------------------------------------------------------
    open(newunit=unit, file=trim(file), access='append')

    !! write convolution initial parameters
    !!--------------------------------------------------------------------------
    write(unit,'("MAXPOOL2D")')
    if(all(this%pool.eq.this%pool(1)))then
       write(unit,'(3X,"POOL_SIZE =",1X,I0)') this%pool(1)
    else
       write(unit,'(3X,"POOL_SIZE =",2(1X,I0))') this%pool
    end if
    if(all(this%strd.eq.this%strd(1)))then
       write(unit,'(3X,"STRIDE =",1X,I0)') this%strd(1)
    else
       write(unit,'(3X,"STRIDE =",2(1X,I0))') this%strd
    end if
    write(unit,'("END MAXPOOL2D")')

    !! close unit
    !!--------------------------------------------------------------------------
    close(unit)

  end subroutine print_maxpool2d
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! forward propagation
!!!#############################################################################
  pure subroutine forward_3d(this, input)
    implicit none
    class(maxpool2d_layer_type), intent(inout) :: this
    real(real12), dimension(:,:,:), intent(in) :: input

    integer :: i, j, m
    integer, dimension(2) :: stride_idx

    
    this%output = 0._real12
    !! perform the pooling operation
    do concurrent(&
         m = 1:this%num_channels,&
         j = 1:this%output_shape(2),&
         i = 1:this%output_shape(1))
       stride_idx = ([i,j] - 1) * this%strd + 1
       this%output(i, j, m) = maxval(&
            input(&
            stride_idx(1):stride_idx(1)+this%pool(1)-1, &
            stride_idx(2):stride_idx(2)+this%pool(2)-1, m))
    end do

  end subroutine forward_3d
!!!#############################################################################


!!!#############################################################################
!!! backward propagation
!!!#############################################################################
  pure subroutine backward_3d(this, input, gradient)
    implicit none
    class(maxpool2d_layer_type), intent(inout) :: this
    real(real12), dimension(:,:,:), intent(in) :: input
    real(real12), &
         dimension(&
         this%output_shape(1),&
         this%output_shape(2),this%num_channels), &
         intent(in) :: gradient

    integer :: i, j, m
    integer, dimension(2) :: stride_idx, max_idx


    this%di = 0._real12
    !! compute gradients for input feature map
    do concurrent(&
         m = 1:this%num_channels,&
         j = 1:this%output_shape(2),&
         i = 1:this%output_shape(1))
       stride_idx = ([i,j] - 1) * this%strd
       !! find the index of the maximum value in the corresponding pooling window
       max_idx = maxloc(input(&
            stride_idx(1)+1:stride_idx(1)+this%pool(1), &
            stride_idx(2)+1:stride_idx(2)+this%pool(2), m))

       !! compute gradients for input feature map
       this%di(&
            stride_idx(1)+max_idx(1), &
            stride_idx(2)+max_idx(2), m) = gradient(i, j, m)

    end do

  end subroutine backward_3d
!!!#############################################################################

end module maxpool2d_layer
!!!#############################################################################
