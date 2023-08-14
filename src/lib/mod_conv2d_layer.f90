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
     integer :: half_x, half_y
     integer :: num_channels
     !integer :: num_channels_out
     integer :: centre_x, centre_y
     integer :: width
     integer :: height
     integer :: num_filters
     real(real12), allocatable, dimension(:) :: bias, bias_incr, db
     real(real12), allocatable, dimension(:,:,:,:) :: weight, weight_incr
     real(real12), allocatable, dimension(:,:,:,:) :: dw
     real(real12), allocatable, dimension(:,:,:) :: output, z
     real(real12), allocatable, dimension(:,:,:) :: di

     class(activation_type), allocatable :: transfer
   contains
     procedure :: forward  => forward_rank
     procedure :: backward => backward_rank
     procedure :: forward_3d
     procedure :: backward_3d
     procedure, pass(this) :: update
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
    layer%half_x   = (layer%kernel_x-1)/2
    layer%half_y   = (layer%kernel_y-1)/2

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
    !! allocate output, weight, and bias shapes
    !!--------------------------------------------------------------------------
    layer%num_channels = input_shape(3)
    !layer%num_channels_out = layer%num_channels * layer%num_filters
    layer%width = floor( (input_shape(2) + 2.0 * layer%pad_y - layer%kernel_y)/&
         real(layer%stride_y) ) + 1
    layer%height = floor( (input_shape(1) + 2.0 * layer%pad_x - layer%kernel_x)/&
         real(layer%stride_x) ) + 1

    allocate(layer%output(layer%height,layer%width,layer%num_filters))
    allocate(layer%z, mold=layer%output)

    allocate(layer%bias(layer%num_filters))

    xend_idx   = layer%half_x + (layer%centre_x - 1)
    yend_idx   = layer%half_y + (layer%centre_y - 1)
    allocate(layer%weight(&
         -layer%half_x:xend_idx,-layer%half_y:yend_idx,&
         layer%num_channels,layer%num_filters))


    !!--------------------------------------------------------------------------
    !! initialise weights and biases steps
    !!--------------------------------------------------------------------------
    allocate(layer%bias_incr, mold=layer%bias)
    allocate(layer%weight_incr, mold=layer%weight)
    layer%bias_incr = 0._real12
    layer%weight_incr = 0._real12


    !!--------------------------------------------------------------------------
    !! initialise gradients
    !!--------------------------------------------------------------------------
    allocate(layer%di(&
         input_shape(1), input_shape(2), input_shape(3)), source=0._real12)
    allocate(layer%dw, mold=layer%weight)
    allocate(layer%db, mold=layer%bias)
    layer%di = 0._real12
    layer%dw = 0._real12
    layer%db = 0._real12


    !!--------------------------------------------------------------------------
    !! initialise kernels and biases
    !!--------------------------------------------------------------------------
    if(present(kernel_initialiser))then
       initialiser_name = kernel_initialiser
    else
       initialiser_name = "he_uniform"
    end if
    allocate(initialiser, source=initialiser_setup(initialiser_name))
    call initialiser%initialise(layer%weight, &
         fan_in=layer%kernel_x*layer%kernel_y+1, fan_out=1)
    if(present(bias_initialiser))then
       initialiser_name = bias_initialiser
    else
       initialiser_name= "zeros"
    end if
    deallocate(initialiser)
    allocate(initialiser, source=initialiser_setup(initialiser_name))
    call initialiser%initialise(layer%bias, &
         fan_in=layer%kernel_x*layer%kernel_y+1, fan_out=1)

  end function layer_setup
!!!#############################################################################


!!!#############################################################################
!!! 
!!!#############################################################################
  pure subroutine forward_3d(this, input)
    implicit none
    class(conv2d_layer_type), intent(inout) :: this
    real(real12), dimension(-this%pad_x+1:,-this%pad_y+1:,:), intent(in) :: input

    integer :: i, j, l, istride, jstride
    integer :: iend_idx, jend_idx, istart, iend, jstart, jend


    !! Perform the convolution operation
    iend_idx = this%half_x + (this%centre_x - 1)
    jend_idx = this%half_y + (this%centre_y - 1)
    do concurrent(i=1:this%height:1, j=1:this%width:1)
       istride = (i-1)*this%stride_x + 1 + (this%pad_x- this%half_x)
       istart  = istride - this%half_x
       iend    = istride + iend_idx
       jstride = (j-1)*this%stride_y + 1 + (this%pad_y- this%half_y)
       jstart  = jstride - this%half_y
       jend    = jstride + jend_idx

       this%z(i,j,:) = this%bias(:)

       do concurrent(l=1:this%num_filters)

          this%z(i,j,l) = this%z(i,j,l) + &
               sum( &                
               input(istart:iend,jstart:jend,:) * &
               this%weight(:,:,:,l) &
               )
       end do

    end do
    
    this%output = this%transfer%activate(this%z) 

  end subroutine forward_3d
!!!#############################################################################


!!!#############################################################################
!!! 
!!!#############################################################################
  pure subroutine backward_3d(this, input, gradient)
    implicit none
    class(conv2d_layer_type), intent(inout) :: this
    real(real12), dimension(-this%pad_x+1:,-this%pad_y+1:,:), intent(in) :: input
    real(real12), dimension(&
         this%height,this%width,this%num_filters), intent(in) :: gradient !was output_gradients

    integer :: l, m, i, j, x, y
    integer :: istride, jstride, ioffset, joffset
    integer :: iend_idx, jend_idx
    integer :: k_x, k_y, int_x, int_y, n_stride_x, n_stride_y
    integer :: i_start, i_end, j_start, j_end
    integer :: x_start, x_end, y_start, y_end
    real(real12), dimension(&
         lbound(this%di,1):ubound(this%di,1),&
         lbound(this%di,2):ubound(this%di,2),this%num_channels) :: di
    real(real12), dimension(this%height,this%width,this%num_filters) :: grad_dz


    !! get size of the input and output feature maps
    k_x = this%kernel_x - 1
    k_y = this%kernel_x - 1
    iend_idx = this%half_x + (this%centre_x - 1)
    jend_idx = this%half_y + (this%centre_y - 1)
    ioffset  = 1 + this%half_x - this%pad_x
    joffset  = 1 + this%half_y - this%pad_y

    int_x = (this%height - 1) * this%stride_x + 1 + iend_idx
    int_y = (this%width  - 1) * this%stride_y + 1 + jend_idx
    n_stride_x = this%height * this%stride_x
    n_stride_y = this%width  * this%stride_y

    !! get gradient multiplied by differential of Z
    grad_dz = 0._real12
    grad_dz(1:this%height,1:this%width,:) = gradient * this%transfer%differentiate(this%z)
    do concurrent(l=1:this%num_filters)
       this%db(l) = this%db(l) + sum(grad_dz(:,:,l))
    end do

    !! Perform the convolution operation
    do concurrent( &
         i=1:this%height:1, &
         j=1:this%width:1, &
         m=1:this%num_channels, &
         l=1:this%num_filters &
         )
       j_start = max(1,           j - this%pad_y)
       j_end   = min(this%width,  j + this%pad_y)
       i_start = max(1,           i - this%pad_x)
       i_end   = min(this%height, i + this%pad_x)

       !! apply convolution to compute weight gradients
       this%dw(:,:,m,l) = this%dw(:,:,m,l) + &
            input(i_start:i_end,j_start:j_end,m) * grad_dz(i_start:i_end,j_start:j_end,l)

    end do


    this%di = 0._real12
    do concurrent( &
         i=1:size(this%di,dim=1):1, &
         j=1:size(this%di,dim=2):1, &
         m=1:this%num_channels, &
         l=1:this%num_filters &
         )

       istride = (i-ioffset)/this%stride_x + 1
       i_start = max(1,           istride)
       i_end   = min(this%height, istride + (i-1)/this%stride_x )
       !! max( ...
       !! ... 1. offset of 1st output idx from centre of krnl     (limit)
       !! ... 2. lowest output idx overlap with leftmost krnl idx (repeating pattern)
       !! ...)
       x_start = max(k_x-i,  -this%half_x + mod(n_stride_x+this%kernel_x-i,this%stride_x))
       !! min( ...
       !! ... 1. offset of last output idx from centre of krnl       (limit)
       !! ... 2. highest output idx overlap with rightmost krnl idx (repeating pattern)
       !! ...)
       x_end   = min(int_x-i, iend_idx    - mod(n_stride_x-1+i,this%stride_x))
       if(x_start.gt.x_end) cycle


       jstride = (j-joffset)/this%stride_y + 1
       j_start = max(1,          jstride)
       j_end   = min(this%width, jstride + (j-1)/this%stride_y )
       !! max( ...
       !! ... 1. offset of 1st output idx from centre of krnl       (limit)
       !! ... 2. lowest output idx overlap with leftmost krnl idx (repeating pattern)
       !! ...)
       y_start = max(k_y-j,  -this%half_y + mod(n_stride_y+this%kernel_y-j,this%stride_y))
       !! min( ...
       !! ... 1. offset of last output idx from centre of krnl      (limit)
       !! ... 2. highest output idx overlap with rightmost krnl idx (repeating pattern)
       !! ...)
       y_end   = min(int_y-j, jend_idx    - mod(n_stride_y-1+j,this%stride_y))
       if(y_start.gt.y_end) cycle


       !! apply full convolution to compute input gradients
       !! https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-8137e4fc2710
       this%di(i,j,m) = &
            this%di(i,j,m) + &
            sum( &
            grad_dz(i_start:i_end:1,j_start:j_end:1,l) * &
            this%weight(x_end:x_start:-this%stride_x,y_end:y_start:-this%stride_y,m,l) )

    end do

    !! all elements of the output are separated by stride_x (stride_y)
    

  end subroutine backward_3d
!!!#############################################################################


!!!#############################################################################
!!!
!!!#############################################################################
  pure subroutine update(this, optimiser, clip)
    use custom_types, only: clip_type
    use optimiser, only: optimiser_type
    use normalisation, only: gradient_clip
    implicit none
    class(conv2d_layer_type), intent(inout) :: this
    type(optimiser_type), intent(in) :: optimiser
    type(clip_type), optional, intent(in) :: clip

    integer :: l


    !! apply gradient clipping
    if(present(clip))then
       do l=1,size(this%bias,dim=1)
          if(clip%l_min_max) call gradient_clip(size(this%dw(:,:,:,l)),&
               this%dw(:,:,:,l),this%db(l),&
               clip_min=clip%min,clip_max=clip%max)
          if(clip%l_norm) &
               call gradient_clip(size(this%dw(:,:,:,l)),&
               this%dw(:,:,:,l),this%db(l),&
               clip_norm=clip%norm)
       end do
    end if

    !! STORE ADAM VALUES IN OPTIMISER

    !! update the convolution layer weights using gradient descent
    call optimiser%optimise(&
         this%weight,&
         this%weight_incr, &
         this%dw)
    !! update the convolution layer bias using gradient descent
    call optimiser%optimise(&
         this%bias,&
         this%bias_incr, &
         this%db)!, &
    !this%bias_m, &
    !this%bias_v)

    this%di = 0._real12
    this%dw = 0._real12
    this%db = 0._real12

  end subroutine update
!!!#############################################################################


end module conv2d_layer
!!!#############################################################################
