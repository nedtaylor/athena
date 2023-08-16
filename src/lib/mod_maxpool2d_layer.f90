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
     integer :: pool_x, pool_y
     integer :: stride_x, stride_y
     integer :: height, width
     integer :: num_channels
     real(real12), allocatable, dimension(:,:,:) :: output
     real(real12), allocatable, dimension(:,:,:) :: di ! gradient of input (i.e. delta)

   contains
     procedure :: forward  => forward_rank
     procedure :: backward => backward_rank
     procedure :: forward_3d
     procedure :: backward_3d
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


!!!#############################################################################
!!!#############################################################################
  pure module function layer_setup( &
       input_shape, &
       pool_size, stride) result(layer)
    implicit none
    integer, dimension(:), intent(in) :: input_shape
    integer, dimension(..), optional, intent(in) :: pool_size
    integer, dimension(..), optional, intent(in) :: stride
    
    type(maxpool2d_layer_type) :: layer

    integer :: xend_idx, yend_idx

    
    !!-----------------------------------------------------------------------
    !! set up pool size
    !!-----------------------------------------------------------------------
    if(present(pool_size))then
       select rank(pool_size)
       rank(0)
          layer%pool_x = pool_size
          layer%pool_y = pool_size
       rank(1)
          layer%pool_x = pool_size(1)
          if(size(pool_size,dim=1).eq.1)then
             layer%pool_y = pool_size(1)
          elseif(size(pool_size,dim=1).eq.2)then
             layer%pool_y = pool_size(2)
          end if
       end select
    else
       layer%pool_x = 2
       layer%pool_y = 2
    end if


    !!-----------------------------------------------------------------------
    !! set up stride
    !!-----------------------------------------------------------------------
    if(present(stride))then
       select rank(stride)
       rank(0)
          layer%stride_x = stride
          layer%stride_y = stride
       rank(1)
          layer%stride_x = stride(1)
          if(size(stride,dim=1).eq.1)then
             layer%stride_y = stride(1)
          elseif(size(stride,dim=1).eq.2)then
             layer%stride_y = stride(2)
          end if
       end select
    else
       layer%stride_x = 2
       layer%stride_y = 2
    end if


    !!-----------------------------------------------------------------------
    !! set up number of channels, width, height
    !!-----------------------------------------------------------------------
    layer%num_channels = input_shape(3)
    layer%width  = floor( (input_shape(2)-layer%pool_y)/real(layer%stride_y) ) + 1
    layer%height = floor( (input_shape(1)-layer%pool_x)/real(layer%stride_x) ) + 1


    !!-----------------------------------------------------------------------
    !! allocate output and gradients
    !!-----------------------------------------------------------------------
    allocate(layer%output(layer%height,layer%width,layer%num_channels), &
         source=0._real12)
    allocate(layer%di(input_shape(1), input_shape(2), input_shape(3)), &
         source=0._real12)

  end function layer_setup
!!!#############################################################################


!!!#############################################################################
!!!#############################################################################
  pure subroutine forward_3d(this, input)
    implicit none
    class(maxpool2d_layer_type), intent(inout) :: this
    real(real12), dimension(:,:,:), intent(in) :: input

    integer :: i, j, m, istride, jstride

    
    this%output = 0._real12
    !! perform the pooling operation
    do m = 1, this%num_channels
       do j = 1, this%width
          jstride = (j-1)*this%stride_y+1
          do i = 1, this%height
             istride = (i-1)*this%stride_x+1
             this%output(i, j, m) = maxval(&
                  input(&
                  istride:istride+this%pool_x-1, &
                  jstride:jstride+this%pool_y-1, m))
          end do
       end do
    end do

  end subroutine forward_3d
!!!#############################################################################


!!!#############################################################################
!!!#############################################################################
  pure subroutine backward_3d(this, input, gradient)
    implicit none
    class(maxpool2d_layer_type), intent(inout) :: this
    real(real12), dimension(:,:,:), intent(in) :: input
    real(real12), dimension(&
         this%height,this%width,this%num_channels), intent(in) :: gradient !was output_gradients
    !! NOTE, gradient is di, not dw

    integer :: i, j, m, istride, jstride
    integer, dimension(2) :: max_index


    this%di = 0._real12
    !! compute gradients for input feature map
    do m = 1, this%num_channels
       do j = 1, this%width
          jstride = (j-1)*this%stride_y
          do i = 1, this%height
             istride = (i-1)*this%stride_x
             !! find the index of the maximum value in the corresponding pooling window
             max_index = maxloc(input(&
                  istride+1:istride+this%pool_x, &
                  jstride+1:jstride+this%pool_y, m))

             !! compute gradients for input feature map
             this%di(&
                  istride+max_index(1), &
                  jstride+max_index(2), m) = gradient(i, j, m)

          end do
       end do
    end do

  end subroutine backward_3d
!!!#############################################################################


end module maxpool2d_layer
!!!#############################################################################
