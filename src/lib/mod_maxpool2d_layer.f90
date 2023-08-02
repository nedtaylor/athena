!!!#############################################################################
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
     procedure :: init
     procedure :: setup
  end type maxpool2d_layer_type



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
  subroutine setup(this, &
       pool_size, stride)
    implicit none
    class(maxpool2d_layer_type), intent(inout) :: this
    integer, dimension(..), optional, intent(in) :: pool_size
    integer, dimension(..), optional, intent(in) :: stride

    if(present(pool_size))then
       select rank(pool_size)
       rank(0)
          this%pool_x = pool_size
          this%pool_y = pool_size
       rank(1)
          this%pool_x = pool_size(1)
          if(size(pool_size,dim=1).eq.1)then
             this%pool_y = pool_size(1)
          elseif(size(pool_size,dim=1).eq.2)then
             this%pool_y = pool_size(2)
          end if
       end select
    else
       this%pool_x = 3
       this%pool_y = 3
    end if

    if(present(stride))then
       select rank(stride)
       rank(0)
          this%stride_x = stride
          this%stride_y = stride
       rank(1)
          this%stride_x = stride(1)
          if(size(stride,dim=1).eq.1)then
             this%stride_y = stride(1)
          elseif(size(stride,dim=1).eq.2)then
             this%stride_y = stride(2)
          end if
       end select
    else
       this%stride_x = 1
       this%stride_y = 1
    end if

  end subroutine setup
!!!#############################################################################


!!!#############################################################################
!!!#############################################################################
  subroutine init(this, input_shape)
    implicit none
    class(maxpool2d_layer_type), intent(inout) :: this
    integer, dimension(:), intent(in) :: input_shape

    integer :: xend_idx, yend_idx

    this%num_channels = input_shape(3)
    this%width  = floor( (input_shape(2)-this%pool_y)/real(this%stride_y) ) + 1
    this%height = floor( (input_shape(1)-this%pool_x)/real(this%stride_x) ) + 1

    allocate(this%output(this%height,this%width,this%num_channels))
    allocate(this%di(input_shape(1), input_shape(2), input_shape(3)))

  end subroutine init
!!!#############################################################################


!!!#############################################################################
!!!#############################################################################
  pure subroutine forward_3d(this, input)
    implicit none
    class(maxpool2d_layer_type), intent(inout) :: this
    real(real12), dimension(:,:,:), intent(in) :: input

    integer :: i, j, m, istride, jstride

    
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


    !! compute gradients for input feature map
    do m = 1, this%num_channels
       do j = 1, this%width
          jstride = (j-1)*this%stride_y+1
          do i = 1, this%height
             istride = (i-1)*this%stride_x+1
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
