!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module conv2d_layer
  use constants, only: real12
  use base_layer, only: base_layer_type
  use custom_types, only: activation_type
  implicit none
  
  
  type, extends(base_layer_type) :: conv2d_layer_type

     integer :: kernel_x, kernel_y
     integer :: stride_x, stride_y
     integer :: pad_y, pad_x
     integer :: num_channels
     integer :: num_channels_out
     integer :: centre_x, centre_y
     integer :: width
     integer :: height
     integer :: num_filters
     real(real12), allocatable, dimension(:) :: bias, db
     real(real12), allocatable, dimension(:,:,:) :: weight, weight_incr
     real(real12), allocatable, dimension(:,:,:) :: output
     real(real12), allocatable, dimension(:,:,:) :: dw, di

     class(activation_type), allocatable :: transfer
   contains
     procedure :: forward  => forward_rank
     procedure :: backward => backward_rank
     procedure :: forward_3d
     procedure :: backward_3d
     procedure :: init
     procedure :: setup
  end type conv2d_layer_type

contains

!!!#############################################################################
!!!#############################################################################
  pure subroutine forward_rank(this, input)
    implicit none
    class(conv2d_layer_type), intent(inout) :: this
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
    class(conv2d_layer_type), intent(inout) :: this
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
       kernel_size, stride, num_filters, activation_scale, activation_function)
    use activation,  only: activation_setup
    implicit none
    class(conv2d_layer_type), intent(inout) :: this
    integer, dimension(..), optional, intent(in) :: kernel_size
    integer, dimension(..), optional, intent(in) :: stride
    integer, optional, intent(in) :: num_filters
    real(real12), optional, intent(in) :: activation_scale
    character(*), optional, intent(in) :: activation_function

    real(real12) :: scale
    character(len=10) :: t_activation_function

    if(present(kernel_size))then
       select rank(kernel_size)
       rank(0)
          this%kernel_x = kernel_size
          this%kernel_y = kernel_size
       rank(1)
          this%kernel_x = kernel_size(1)
          if(size(kernel_size,dim=1).eq.1)then
             this%kernel_y = kernel_size(1)
          elseif(size(kernel_size,dim=1).eq.2)then
             this%kernel_y = kernel_size(2)
          end if
       end select
    else
       this%kernel_x = 3
       this%kernel_y = 3
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

    if(present(num_filters))then
       this%num_filters = num_filters
    end if
    
    !!--------------------------------------------------------------------------
    !! set activation and derivative functions based on input name
    !!--------------------------------------------------------------------------
    if(present(activation_function))then
       t_activation_function = activation_function
    else
       t_activation_function = "none"
    end if
    if(present(activation_scale))then
       scale = activation_scale
    else
       scale = 1._real12
    end if
       
    allocate(this%transfer, source=activation_setup(t_activation_function, scale))



  end subroutine setup
!!!#############################################################################


!!!#############################################################################
!!!#############################################################################
  subroutine init(this, input_shape)
    implicit none
    class(conv2d_layer_type), intent(inout) :: this
    integer, dimension(:), intent(in) :: input_shape

    integer :: xend_idx, yend_idx

    this%num_channels = input_shape(3)
    this%width = floor( (input_shape(2) + 2.0 * this%pad_y - this%kernel_y)/&
         real(this%stride_y) ) + 1
    this%height = floor( (input_shape(1) + 2.0 * this%pad_x - this%kernel_x)/&
         real(this%stride_x) ) + 1

    allocate(this%output(this%height,this%width,this%num_channels_out))
    allocate(this%bias(this%num_filters))
    xend_idx   = this%pad_x + (this%centre_x - 1)
    yend_idx   = this%pad_y + (this%centre_y - 1)
    allocate(this%weight_incr(&
         -this%pad_x:xend_idx,-this%pad_y:yend_idx,this%num_filters))
    this%weight_incr = 0._real12

    allocate(this%weight(&
         -this%pad_x:xend_idx,-this%pad_y:yend_idx,this%num_filters))

  end subroutine init
!!!#############################################################################


!!!#############################################################################
!!!#############################################################################
  pure subroutine forward_3d(this, input)
    implicit none
    class(conv2d_layer_type), intent(inout) :: this
    real(real12), dimension(-this%pad_x:,-this%pad_y:,:), intent(in) :: input

    integer :: i, j, l, m, ichannel, istride, jstride
    integer :: iend_idx, jend_idx, jstart, jend


    !! get size of the input and output feature maps

    !! Perform the convolution operation
    ichannel = 0
    iend_idx = this%pad_x + (this%centre_x - 1)
    jend_idx = this%pad_y + (this%centre_y - 1)
    do l=1,this%num_filters
       do m=1,this%num_channels
          ichannel = ichannel + 1
          this%output(i,j,ichannel) = this%bias(l)

          !! end_stride is the same as output_size
          !! ... hence, forward does not need the fix
          do j=1,this%width,1
             jstride = (j-1)*this%stride_y + 1
             jstart = jstride - this%pad_y
             jend   = jstride + jend_idx
             do i=1,this%height,1
                istride = (i-1)*this%stride_x + 1
          
                this%output(i,j,ichannel) = this%output(i,j,ichannel) + &
                     sum( &                
                     input(&
                     istride-this%pad_x:istride+iend_idx,&
                     jstart:jend,m) * &
                     this%weight(:,:,l) &
                )
          
             end do
          end do

       end do
    end do
    
    this%output = this%transfer%activate(this%output) 

  end subroutine forward_3d
!!!#############################################################################


!!!#############################################################################
!!!#############################################################################
  pure subroutine backward_3d(this, input, gradient)
    implicit none
    class(conv2d_layer_type), intent(inout) :: this
    real(real12), dimension(-this%pad_x:,-this%pad_y:,:), intent(in) :: input
    real(real12), dimension(&
         this%height,this%width,this%num_channels_out), intent(in) :: gradient !was output_gradients

    integer :: ichannel
    integer :: l, m, i, j, x, y
    integer :: istride, jstride
    integer :: iend_idx, jend_idx, iup_idx, jup_idx
    integer :: i_start, i_end, j_start, j_end
    integer :: x_start, x_end, y_start, y_end
    real(real12), dimension(1) :: bias_diff
    real(real12), dimension(&
         lbound(this%di,1):ubound(this%di,1),&
         lbound(this%di,2):ubound(this%di,2),this%num_channels) :: di


    !! get size of the input and output feature maps
    bias_diff = this%transfer%differentiate([1._real12])

    iend_idx = this%pad_x + (this%centre_x - 1)
    jend_idx = this%pad_y + (this%centre_y - 1)

    iup_idx = ubound(input, dim=1) - this%kernel_x + 1 + this%pad_x
    jup_idx = ubound(input, dim=2) - this%kernel_y + 1 + this%pad_y


    !! Perform the convolution operation
    ichannel = 0
    this%di = 0._real12
    this%dw = 0._real12
    this%db = 0._real12
    do l=1,this%num_filters
       do m=1,this%num_channels
          ichannel = ichannel + 1
       
          !! https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/
          !!https://www.youtube.com/watch?v=pUCCd2-17vI
       
          !! apply full convolution to compute input gradients
          i_input_loop: do j=1,this%width
             j_start = max(1,              j - this%pad_y)
             j_end   = min(this%width,     j + jend_idx)
             y_start = max(1 - j,         -jend_idx)
             y_end   = min(this%width - j, this%pad_y)
             jstride = (j - 1)*this%stride_y + 1

             j_input_loop: do i=1,this%height
                i_start = max(1,               i - this%pad_x)
                i_end   = min(this%height,     i + iend_idx)
                x_start = max(1 - i,          -iend_idx)
                x_end   = min(this%height - i, this%pad_x)
                istride = (i - 1)*this%stride_x + 1

                di(istride,jstride,m) = &
                     sum( &
                     gradient(&
                     i_start:i_end,&
                     j_start:j_end,ichannel) * &
                     this%weight(x_start:x_end,y_start:y_end,l) &
                     )

       
             end do j_input_loop
          end do i_input_loop
       
          !! apply convolution to compute weight gradients
          y_weight_loop: do y=-this%pad_y,jend_idx,1
             x_weight_loop: do x=-this%pad_x,iend_idx,1
                this%dw(x,y,l) = this%dw(x,y,l) + &
                     sum(gradient(:,:,ichannel) * &
                     input(&
                     x+1:iup_idx+x:this%stride_x,&
                     y+1:jup_idx+y:this%stride_y,m))
             end do x_weight_loop
          end do y_weight_loop
       
          !! compute gradients for bias
          !! https://stackoverflow.com/questions/58036461/how-do-you-calculate-the-gradient-of-bias-in-a-conolutional-neural-networo
          !! https://saturncloud.io/blog/how-to-calculate-the-gradient-of-bias-in-a-convolutional-neural-network/
          this%db(l) = this%db(l) + &
               sum(gradient(:,:,ichannel)) * bias_diff(1)
       
       end do
       this%di(:,:,l) = sum(di(:,:,:) * &
            this%transfer%differentiate(input(&
            1:size(this%di,1):this%stride_x,&
            1:size(this%di,2):this%stride_y,:)),dim=3)
    end do
    
  end subroutine backward_3d
!!!#############################################################################


end module conv2d_layer
!!!#############################################################################
