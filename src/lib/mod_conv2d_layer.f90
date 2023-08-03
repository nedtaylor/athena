!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module conv2d_layer
  use constants, only: real12
  use base_layer, only: base_layer_type
  use custom_types, only: activation_type, initialiser_type
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
  end type conv2d_layer_type

  
  interface conv2d_layer_type
     module function layer_setup( &
          input_shape, &
          num_filters, kernel_size, stride, padding, &
          activation_function, activation_scale, &
          kernel_initialiser, bias_initialiser) result(layer)
       integer, dimension(:), intent(in) :: input_shape
       integer, optional, intent(in) :: num_filters
       integer, dimension(..), optional, intent(in) :: kernel_size
       integer, dimension(..), optional, intent(in) :: stride
       real(real12), optional, intent(in) :: activation_scale
       character(*), optional, intent(in) :: activation_function, &
            kernel_initialiser, bias_initialiser, padding
       type(conv2d_layer_type) :: layer
     end function layer_setup
  end interface conv2d_layer_type


  private
  public :: conv2d_layer_type


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
  module function layer_setup( &
       input_shape, &
       num_filters, kernel_size, stride, padding, &
       activation_function, activation_scale, &
       kernel_initialiser, bias_initialiser) result(layer)
    !! add in clipping/constraint options
    !! add in dilation
    !! add in padding handler
    use activation,  only: activation_setup
    use initialiser, only: initialiser_setup
    use misc_ml, only: set_padding
    implicit none
    integer, dimension(:), intent(in) :: input_shape
    integer, optional, intent(in) :: num_filters
    integer, dimension(..), optional, intent(in) :: kernel_size
    integer, dimension(..), optional, intent(in) :: stride
    real(real12), optional, intent(in) :: activation_scale
    character(*), optional, intent(in) :: activation_function, &
         kernel_initialiser, bias_initialiser, padding

    type(conv2d_layer_type) :: layer

    integer :: xend_idx, yend_idx
    real(real12) :: scale
    character(len=10) :: t_activation_function, initialiser_name
    character(len=20) :: t_padding
    class(initialiser_type), allocatable :: initialiser



    !!-----------------------------------------------------------------------
    !! set up number of filters
    !!-----------------------------------------------------------------------
    if(present(num_filters))then
       layer%num_filters = num_filters
    else
       layer%num_filters = 32
    end if
    
    
    !!-----------------------------------------------------------------------
    !! set up kernel size
    !!-----------------------------------------------------------------------
    if(present(kernel_size))then
       select rank(kernel_size)
       rank(0)
          layer%kernel_x = kernel_size
          layer%kernel_y = kernel_size
       rank(1)
          layer%kernel_x = kernel_size(1)
          if(size(kernel_size,dim=1).eq.1)then
             layer%kernel_y = kernel_size(1)
          elseif(size(kernel_size,dim=1).eq.2)then
             layer%kernel_y = kernel_size(2)
          end if
       end select
    else
       layer%kernel_x = 3
       layer%kernel_y = 3
    end if
    !! odd or even kernel/filter size
    !!-----------------------------------------------------------------------
    layer%centre_x = 2 - mod(layer%kernel_x, 2)
    layer%centre_y = 2 - mod(layer%kernel_y, 2)

    if(present(padding))then
       t_padding = padding
    else
       t_padding = "valid"
    end if
    call set_padding(layer%pad_x, layer%kernel_x, t_padding)
    call set_padding(layer%pad_y, layer%kernel_y, t_padding)


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
       layer%stride_x = 1
       layer%stride_y = 1
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
       
    allocate(layer%transfer, source=activation_setup(t_activation_function, scale))


    !!--------------------------------------------------------------------------
    !! set up output, weight, and bias shapes
    !!--------------------------------------------------------------------------
    layer%num_channels = input_shape(3)
    layer%width = floor( (input_shape(2) + 2.0 * layer%pad_y - layer%kernel_y)/&
         real(layer%stride_y) ) + 1
    layer%height = floor( (input_shape(1) + 2.0 * layer%pad_x - layer%kernel_x)/&
         real(layer%stride_x) ) + 1

    allocate(layer%output(layer%height,layer%width,layer%num_channels_out))
    allocate(layer%bias(layer%num_filters))
    xend_idx   = layer%pad_x + (layer%centre_x - 1)
    yend_idx   = layer%pad_y + (layer%centre_y - 1)
    allocate(layer%weight_incr(&
         -layer%pad_x:xend_idx,-layer%pad_y:yend_idx,layer%num_filters))
    layer%weight_incr = 0._real12

    allocate(layer%weight(&
         -layer%pad_x:xend_idx,-layer%pad_y:yend_idx,layer%num_filters))
    

    !!--------------------------------------------------------------------------
    !! initialise kernels and biases
    !!--------------------------------------------------------------------------
    if(present(kernel_initialiser))then
       initialiser_name = kernel_initialiser
    else
       initialiser_name = "he_uniform"
    end if
    initialiser = initialiser_setup(initialiser_name)
    call initialiser%initialise(layer%weight, &
         fan_in=layer%kernel_x*layer%kernel_y+1, fan_out=1)
    if(present(bias_initialiser))then
       initialiser_name = bias_initialiser
    else
       initialiser_name= "zeros"
    end if
    initialiser = initialiser_setup(initialiser_name)
    call initialiser%initialise(layer%bias, &
         fan_in=layer%kernel_x*layer%kernel_y+1, fan_out=1)

  end function layer_setup
!!!#############################################################################


!!!#############################################################################
!!!#############################################################################
  pure subroutine forward_3d(this, input)
    implicit none
    class(conv2d_layer_type), intent(inout) :: this
    real(real12), dimension(-this%pad_x+1:,-this%pad_y+1:,:), intent(in) :: input

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
    real(real12), dimension(-this%pad_x+1:,-this%pad_y+1:,:), intent(in) :: input
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
