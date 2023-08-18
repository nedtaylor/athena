!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module maxpool3d_layer
  use constants, only: real12
  use base_layer, only: base_layer_type
  implicit none
  
  
  type, extends(base_layer_type) :: maxpool3d_layer_type
     !! strd = stride (step)
     !! pool = pool
     integer, dimension(3) :: pool, strd
     integer :: num_channels
     real(real12), allocatable, dimension(:,:,:,:) :: output
     real(real12), allocatable, dimension(:,:,:,:) :: di ! gradient of input (i.e. delta)
   contains
     procedure :: forward  => forward_rank
     procedure :: backward => backward_rank
     procedure :: forward_4d
     procedure :: backward_4d
  end type maxpool3d_layer_type

  
  interface maxpool3d_layer_type
     pure module function layer_setup( &
          input_shape, &
          pool_size, stride) result(layer)
       integer, dimension(:), intent(in) :: input_shape
       integer, dimension(..), optional, intent(in) :: pool_size
       integer, dimension(..), optional, intent(in) :: stride
       type(maxpool3d_layer_type) :: layer
     end function layer_setup
  end interface maxpool3d_layer_type


  private
  public :: maxpool3d_layer_type


contains

!!!#############################################################################
!!!#############################################################################
  pure subroutine forward_rank(this, input)
    implicit none
    class(maxpool3d_layer_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: input

    select rank(input); rank(4)
       call forward_4d(this, input)
    end select
  end subroutine forward_rank
!!!#############################################################################


!!!#############################################################################
!!!#############################################################################
  pure subroutine backward_rank(this, input, gradient)
    implicit none
    class(maxpool3d_layer_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: input
    real(real12), dimension(..), intent(in) :: gradient

    select rank(input); rank(4)
    select rank(gradient); rank(4)
      call backward_4d(this, input, gradient)
    end select
    end select
  end subroutine backward_rank
!!!#############################################################################


!!!#############################################################################
!!!#############################################################################
  pure module function layer_setup( &
       input_shape, &
       pool_size, stride) result(layer)
    implicit none
    integer, dimension(:), intent(in) :: input_shape
    integer, dimension(..), optional, intent(in) :: pool_size
    integer, dimension(..), optional, intent(in) :: stride
    
    type(maxpool3d_layer_type) :: layer

    integer, dimension(3) :: end_idx

    
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
             layer%pool(2:) = pool_size(1)
          elseif(size(pool_size,dim=1).eq.3)then
             layer%pool(2:) = pool_size(2:)
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
             layer%strd(2:) = stride(1)
          elseif(size(stride,dim=1).eq.2)then
             layer%strd(2:) = stride(2:)
          end if
       end select
    else
       layer%strd = 2
    end if


    !!-----------------------------------------------------------------------
    !! set up number of channels, width, height
    !!-----------------------------------------------------------------------
    layer%num_channels = input_shape(4)
    layer%input_shape  = input_shape(:4)
    allocate(layer%output_shape(4))
    layer%output_shape(:3) = &
         floor( (input_shape(:3)-layer%pool)/real(layer%strd)) + 1
    layer%output_shape(4) = input_shape(4)
    

    !!-----------------------------------------------------------------------
    !! allocate output and gradients
    !!-----------------------------------------------------------------------
    allocate(layer%output(&
         layer%output_shape(1),&
         layer%output_shape(2),&
         layer%output_shape(3),layer%num_channels), &
         source=0._real12)
    allocate(layer%di(&
         input_shape(1),&
         input_shape(2),&
         input_shape(3), input_shape(4)), &
         source=0._real12)

  end function layer_setup
!!!#############################################################################


!!!#############################################################################
!!!#############################################################################
  pure subroutine forward_4d(this, input)
    implicit none
    class(maxpool3d_layer_type), intent(inout) :: this
    real(real12), dimension(:,:,:,:), intent(in) :: input

    integer :: i, j, k, m
    integer, dimension(3) :: stride_idx


    this%output = 0._real12
    !! perform the pooling operation
    do concurrent(&
         m = 1:this%num_channels,&
         k = 1:this%output_shape(3),&
         j = 1:this%output_shape(2),&
         i = 1:this%output_shape(1))
       stride_idx = ([i,j,k] - 1) * this%strd + 1
       this%output(i, j, k, m) = maxval(&
            input(&
            stride_idx(1):stride_idx(1)+this%pool(1)-1, &
            stride_idx(2):stride_idx(2)+this%pool(2)-1, &
            stride_idx(3):stride_idx(3)+this%pool(3)-1, m))
    end do

  end subroutine forward_4d
!!!#############################################################################


!!!#############################################################################
!!!#############################################################################
  pure subroutine backward_4d(this, input, gradient)
    implicit none
    class(maxpool3d_layer_type), intent(inout) :: this
    real(real12), dimension(:,:,:,:), intent(in) :: input
    real(real12), dimension(&
         this%output_shape(1),&
         this%output_shape(2),&
         this%output_shape(3),this%num_channels), intent(in) :: gradient

    integer :: i, j, k, m
    integer, dimension(3) :: stride_idx, max_idx


    this%di = 0._real12
    !! compute gradients for input feature map
    do concurrent(&
         m = 1:this%num_channels,&
         k = 1:this%output_shape(3),&
         j = 1:this%output_shape(2),&
         i = 1:this%output_shape(1))
       stride_idx = ([i,j,k] - 1) * this%strd
       !! find the index of the maximum value in the corresponding pooling window
       max_idx = maxloc(input(&
            stride_idx(1)+1:stride_idx(1)+this%pool(1), &
            stride_idx(2)+1:stride_idx(2)+this%pool(2), &
            stride_idx(3)+1:stride_idx(3)+this%pool(3), m))

       !! compute gradients for input feature map
       this%di(&
            stride_idx+max_idx(1), &
            stride_idx+max_idx(2), &
            stride_idx+max_idx(3), m) = gradient(i, j, k, m)

    end do

  end subroutine backward_4d
!!!#############################################################################


end module maxpool3d_layer
!!!#############################################################################
