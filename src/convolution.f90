!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module ConvolutionLayer
  use constants, only: real12
  use random, only: random_setup
  use custom_types, only: clip_type, convolution_type, activation_type, &
       initialiser_type, learning_parameters_type
  use misc_ml, only: get_padding_half, update_weight
  use activation,  only: activation_setup
  use initialiser, only: initialiser_setup
  implicit none


  !! https://www.nag.com/nagware/np/r62_doc/manual/compiler_9_2.html
  !! https://www.intel.com/content/www/us/en/docs/fortran-compiler/developer-guide-reference/2023-0/generic.html
  type gradient_type
     real(real12), allocatable, dimension(:,:) :: delta
     real(real12), allocatable, dimension(:,:) :: weight
     real(real12), allocatable, dimension(:,:) :: m
     real(real12), allocatable, dimension(:,:) :: v
     real(real12) :: bias
     real(real12) :: bias_m
     real(real12) :: bias_v
   contains
     procedure :: add_t_t => gradient_add
     generic :: operator(+) => add_t_t !, public
  end type gradient_type

  class(activation_type), allocatable :: transfer!activation


  integer :: padding_lw

  type(convolution_type), allocatable, dimension(:) :: convolution
  type(learning_parameters_type) :: adaptive_parameters

!!! NEED TO MAKE A TYPE TO HANDLE THE OUTPUT OF EACH CONVOLUTION LAYER
!!! THIS IS BECAUSE, if we use different stride for ...
!!! ... each layer, then the dimensions of the output arrays will be ...
!!! ... different for each layer

  private

  public :: convolution
  public :: gradient_type
  public :: allocate_gradients
  public :: initialise_gradients

  public :: initialise, forward, backward
  public :: update_weights_and_biases
  public :: write_file
  


contains

!!!#############################################################################
!!! gradient derived type addition
!!!#############################################################################
  elemental function gradient_add(a,b) result(output)
    implicit none
    class(gradient_type), intent(in) :: a,b
    type(gradient_type) :: output
  
    allocate(output%weight,mold=a%weight)
    output%weight = a%weight + b%weight
    output%bias  = a%bias + b%bias
    output%delta = a%delta + b%delta
    if(allocated(a%m))then
       output%m = a%m !+ input%m
       output%bias_m = a%bias_m
    end if
    if(allocated(a%v))then
       output%v = a%v !+ input%v
       output%bias_v = a%bias_v
    end if
        
  end function gradient_add
!!!#############################################################################


!!!#############################################################################
!!! print weights
!!!#############################################################################
  subroutine cv_print_weights(unit)
    implicit none
    integer, intent(in) :: unit
    integer :: l

    do l=1,size(convolution,1)
       write(unit,*) convolution(l)%weight
    end do

  end subroutine cv_print_weights
!!!#############################################################################


!!!#############################################################################
!!! initialise convolutional layer
!!!#############################################################################
  subroutine initialise(seed, num_layers, kernel_size, stride, &
       learning_parameters, file, &
       full_padding, activation_function, activation_scale, &
       kernel_initialiser, bias_initialiser)
    implicit none
    integer, intent(in), optional :: seed
    integer, intent(in), optional :: num_layers
    real(real12), optional, intent(in) :: activation_scale
    integer, dimension(:), intent(in), optional :: kernel_size, stride
    character(*), optional, intent(in) :: file, activation_function, &
         kernel_initialiser, bias_initialiser
    logical, optional, intent(in) :: full_padding
    type(learning_parameters_type), optional, intent(in) :: learning_parameters

    integer :: l
    integer :: itmp1,itmp2,nseed
    integer :: start_idx, end_idx
    real(real12) :: scale
    logical :: t_full_padding
    character(len=10) :: t_activation_function
    class(initialiser_type), allocatable :: kernel_init, bias_init
    integer, allocatable, dimension(:) :: seed_arr
    character(:), allocatable :: t_kernel_initialiser, t_bias_initialiser

    
!!!! num_layers has taken over for output_channels (or cv_num_filters)

    !!--------------------------------------------------------------------------
    !! set defaults if not present
    !!--------------------------------------------------------------------------
    if(present(full_padding))then
       t_full_padding = full_padding
    else
       t_full_padding = .false.
    end if
    if(present(kernel_initialiser))then
       t_kernel_initialiser = kernel_initialiser
    else
       t_kernel_initialiser = "he_uniform"
    end if
    if(present(bias_initialiser))then
       t_bias_initialiser = bias_initialiser
    else
       t_bias_initialiser = "zeros"
    end if
    write(*,'("CV kernel initialiser: ",A)') t_kernel_initialiser
    write(*,'("CV bias initialiser: ",A)')   t_bias_initialiser


    !! if file, read in weights and biases
    !! ... if no file is given, weights and biases to a default
    if(present(file))then
       !!-----------------------------------------------------------------------
       !! read convolution layer data from file
       !!-----------------------------------------------------------------------
       call read_file(file)
       return
    elseif(present(num_layers).and.present(kernel_size).and.present(stride))then
       !!-----------------------------------------------------------------------
       !! initialise random seed
       !!-----------------------------------------------------------------------
       if(present(seed))then
          call random_setup(seed, num_seed=1, restart=.false.)
       else
          call random_setup(num_seed=1, restart=.false.)
       end if

       !!-----------------------------------------------------------------------
       !! randomly initialise convolution layers
       !!-----------------------------------------------------------------------
       allocate(convolution(num_layers))
       itmp1 = kernel_size(1)
       itmp2 = stride(1)
       do l=1,num_layers
          if(size(kernel_size,dim=1).gt.1) itmp1 = kernel_size(l)
          if(size(stride,dim=1).gt.1)      itmp2 = stride(l)
          convolution(l)%kernel_size = itmp1
          convolution(l)%stride      = itmp2
          
          !! padding width
          if(t_full_padding)then
             convolution(l)%pad  = itmp1 - 1
          else
             convolution(l)%pad = get_padding_half(itmp1)
          end if
       
          !! odd or even kernel/filter size
          convolution(l)%centre_width = 2 - mod(itmp1,2)
       
          start_idx = -convolution(l)%pad
          end_idx   = convolution(l)%pad + (convolution(l)%centre_width - 1)
          allocate(convolution(l)%weight(start_idx:end_idx,start_idx:end_idx))
          call random_number(convolution(l)%bias)
          call random_number(convolution(l)%weight)
          allocate(convolution(l)%weight_incr(start_idx:end_idx,start_idx:end_idx))
          convolution(l)%weight_incr(:,:) = 0._real12


          !! determine initialisation method and initialise accordingly
          !!--------------------------------------------------------------------
          kernel_init = initialiser_setup(t_kernel_initialiser)
          call kernel_init%initialise(convolution(l)%weight, itmp1*itmp1+1, 1)
          bias_init = initialiser_setup(t_bias_initialiser)
          call bias_init%initialise(convolution(l)%bias, itmp1*itmp1+1, 1)

       end do
    else
       write(0,*) "ERROR: Not enough optional arguments provided to initialse CV"
       write(0,*) "Either provide (file) or (num_layers, kernel_size, and stride)"
       write(0,*) "... seed is also optional for the latter set)"
       write(0,*) "Exiting..."
       stop
    end if


    !!-----------------------------------------------------------------------
    !! get lower padding width
    !!-----------------------------------------------------------------------
    padding_lw = -maxval(convolution(:)%pad) + 1
    

    !!-----------------------------------------------------------------------
    !! set activation and derivative functions based on input name
    !!-----------------------------------------------------------------------
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
    transfer = activation_setup(t_activation_function, scale)
 

    !!-----------------------------------------------------------------------
    !! set learning parameters
    !!-----------------------------------------------------------------------
    if(present(learning_parameters))then
       adaptive_parameters = learning_parameters
    else
       adaptive_parameters%method = "none"
    end if



  end subroutine initialise
!!!#############################################################################


!!!#############################################################################
!!! allocate gradient derived type
!!!#############################################################################
  subroutine allocate_gradients(gradients, mold, reallocate)
    implicit none
    type(gradient_type), dimension(:), intent(in) :: mold
    type(gradient_type), allocatable, dimension(:), intent(inout) :: gradients
    logical, optional, intent(in) :: reallocate
    integer :: l, start_idx, end_idx
    integer :: num_filters, input_size
    logical :: t_reallocate
    

    if(present(reallocate))then
       t_reallocate = reallocate
    else
       t_reallocate = .true.
    end if

    num_filters = size(mold,dim=1)
    if(.not.allocated(gradients).or.t_reallocate)then
       input_size = size(mold(1)%delta,dim=1)
       if(allocated(gradients)) deallocate(gradients)
       allocate(gradients(num_filters))
       do l=1,num_filters
          start_idx = -convolution(l)%pad
          end_idx   = convolution(l)%pad + (convolution(l)%centre_width - 1)
          allocate(gradients(l)%weight(start_idx:end_idx,start_idx:end_idx))
          allocate(gradients(l)%delta(input_size,input_size))
          gradients(l)%weight = 0._real12
          gradients(l)%bias = 0._real12
          gradients(l)%delta = 0._real12

          if(allocated(mold(l)%m))then
             if(allocated(gradients(l)%m)) deallocate(gradients(l)%m)
             if(allocated(gradients(l)%v)) deallocate(gradients(l)%v)
             allocate(gradients(l)%m(start_idx:end_idx,start_idx:end_idx))
             allocate(gradients(l)%v(start_idx:end_idx,start_idx:end_idx))
             gradients(l)%m = 0._real12
             gradients(l)%v = 0._real12
             gradients(l)%bias_m = 0._real12
             gradients(l)%bias_v = 0._real12
          end if

       end do
    else
       do l=1,num_filters
          gradients(l)%weight = 0._real12
          gradients(l)%bias = 0._real12
          gradients(l)%delta = 0._real12
          if(allocated(gradients(l)%m))then
             gradients(l)%m = 0._real12
             gradients(l)%v = 0._real12
             gradients(l)%bias_m = 0._real12
             gradients(l)%bias_v = 0._real12
          end if
       end do
    end if

    return
  end subroutine allocate_gradients
!!!#############################################################################


!!!#############################################################################
!!! initialise gradient derived type
!!!#############################################################################
  subroutine initialise_gradients(gradients, input_size, adam_learning)
    implicit none
    integer, intent(in) :: input_size
    type(gradient_type), allocatable, dimension(:), intent(out) :: gradients
    logical, optional, intent(in) :: adam_learning
    
    integer :: l, start_idx, end_idx


    if(allocated(gradients)) deallocate(gradients)
    allocate(gradients(size(convolution,1)))
    do l=1,size(convolution,1)
       start_idx = -convolution(l)%pad
       end_idx   = convolution(l)%pad + (convolution(l)%centre_width - 1)
       allocate(gradients(l)%weight(start_idx:end_idx,start_idx:end_idx))
       allocate(gradients(l)%delta(input_size,input_size))
       gradients(l)%weight = 0._real12
       gradients(l)%bias = 0._real12
       gradients(l)%delta = 0._real12
    
       if(present(adam_learning))then
          if(adam_learning)then
             if(allocated(gradients(l)%m)) deallocate(gradients(l)%m)
             if(allocated(gradients(l)%v)) deallocate(gradients(l)%v)
             allocate(gradients(l)%m(start_idx:end_idx,start_idx:end_idx))
             allocate(gradients(l)%v(start_idx:end_idx,start_idx:end_idx))
             gradients(l)%m = 0._real12
             gradients(l)%v = 0._real12
             gradients(l)%bias_m = 0._real12
             gradients(l)%bias_v = 0._real12
          end if
       end if

    end do

    return
  end subroutine initialise_gradients
!!!#############################################################################


!!!#############################################################################
!!! read convolutional layer from save file
!!!#############################################################################
  subroutine read_file(file)
    use misc, only: to_lower
    use infile_tools, only: assign_val, assign_vec
    implicit none
    character(*), intent(in) :: file

    integer :: i, j, l, istart, istart_weights, itmp1
    integer :: start_idx, end_idx, num_filters
    integer :: unit, stat
    real(real12) :: activation_scale
    character(:), allocatable :: activation_function
    character(:), allocatable :: padding_type
    character(6) :: line_no
    character(1024) :: buffer, tag
    logical :: found, found_num_filters


    found_num_filters = .false.
    if(len(trim(file)).gt.0)then
       unit = 10
       found = .false.
       open(unit, file=trim(file))
       i = 0

       !! check for start of convolution card
       card_check: do while (.not.found)
          i = i + 1
          read(unit,'(A)',iostat=stat) buffer
          if(stat.ne.0)then
             write(0,*) "ERROR: file hit error (EoF?) before CONVOLUTION section"
             write(0,*) "Exiting..."
             stop
          end if
          if(trim(adjustl(buffer)).eq."CONVOLUTION")then
             istart = i
             found = .true.
          end if
       end do card_check

       !! loop over tags in convolution card
       i = istart
       tag_loop: do
          i = i + 1

          !! check for end of file
          read(unit,'(A)',iostat=stat) buffer
          if(stat.ne.0)then
             write(0,*) "ERROR: file hit error (EoF?) before encountering END CONVOLUTION"
             write(0,*) "Exiting..."
             stop
          end if
          found = .false.

          !! check for end of convolution card
          if(trim(adjustl(buffer)).ne."END CONVOLUTION")then
             exit tag_loop
          end if
          
          tag=trim(adjustl(buffer))
          if(scan(buffer,"=").ne.0) tag=trim(tag(:scan(tag,"=")-1))

          !! read number of filters from save file
          select case(trim(tag))
          case("NUM_FILTERS")
             if(.not.found_num_filters)then
                call assign_val(buffer, num_filters, itmp1)
                found_num_filters = .true.
                allocate(convolution(num_filters))
                rewind(unit)
                do j=1,istart
                   read(unit,*)
                end do
                i = istart
                 cycle tag_loop
             end if
          case default
             if(.not.found_num_filters) cycle tag_loop
          end select

          !! read parameters from save file
          select case(trim(tag))
          case("ACTIVATION_FUNCTION")
             call assign_val(buffer, activation_function, itmp1)
          case("ACTIVATION_SCALE")
             call assign_val(buffer, activation_scale, itmp1)
          case("PADDING_TYPE")
             call assign_val(buffer, padding_type, itmp1)
             padding_type = to_lower(padding_type)
          case("KERNEL_SIZE")
             call assign_vec(buffer, convolution(:)%kernel_size, itmp1)
          case("STRIDE")
             call assign_vec(buffer, convolution(:)%stride, itmp1)
          case("BIAS")
             call assign_vec(buffer, convolution(:)%bias, itmp1)
          case("WEIGHTS")
             istart_weights = i
             cycle tag_loop
          case default
             write(*,*) "Unrecognised line in cnn input file"
             stop 0
          end select
       end do tag_loop

       !! set transfer activation function
       transfer = activation_setup(activation_function, activation_scale)

       !! rewind file to WEIGHTS tag
       rewind(unit)
       do j=1,istart_weights
          read(unit,*)
       end do

       !! allocate convolutional layer and read weights
       do l=1,num_filters
          !! padding width
          if(padding_type.eq."full")then
             convolution(l)%pad  = itmp1 - 1
          elseif(padding_type.eq."none".or.padding_type.eq."valid")then
             convolution(l)%pad = 0
          else
             convolution(l)%pad = get_padding_half(convolution(l)%kernel_size)
          end if
          !! odd or even kernel/filter size
          convolution(l)%centre_width = 2 - mod(convolution(l)%kernel_size,2)
          start_idx = -convolution(l)%pad
          end_idx   = convolution(l)%pad + (convolution(l)%centre_width - 1)
          allocate(convolution(l)%weight(start_idx:end_idx,start_idx:end_idx))
          allocate(convolution(l)%weight_incr(start_idx:end_idx,start_idx:end_idx))
          convolution(l)%bias_incr = 0._real12
          convolution(l)%weight_incr(:,:) = 0._real12
          read(unit,'(A)') buffer
          read(buffer,*) convolution(l)%weight
       end do

       !! check for end of weights card
       read(unit,'(A)') buffer
       if(trim(adjustl(buffer)).ne."END WEIGHTS")then
          write(line_no,'(I0)') num_filters + istart_weights + 1
          stop "ERROR: END WEIGHTS not where expected, line"//trim(line_no)
       end if
       close(unit)

       return
    end if

  end subroutine read_file
!!!#############################################################################


!!!#############################################################################
!!! write convolutional layer to save file
!!!#############################################################################
  subroutine write_file(file)
    implicit none
    character(*), intent(in) :: file

    integer :: num_layers
    integer :: l
    integer :: unit
    character(128) :: fmt

    unit = 10
    
    open(unit, file=trim(file), access='append')

    num_layers = size(convolution,dim=1)
    write(unit,'("CONVOLUTION")')
    write(unit,'(3X,"ACTIVATION_FUNCTION = ",A)') transfer%name
    write(unit,'(3X,"ACTIVATION_SCALE = ",F0.9)') transfer%scale
    write(unit,'(3X,"NUM_LAYERS = ",I0)') size(convolution,dim=1)

    write(fmt,'("(3X,""STRIDE ="",",I0,"(1X,I0))")') num_layers
    write(unit,trim(fmt)) convolution(:)%kernel_size

    write(fmt,'("(3X,""STRIDE ="",",I0,"(1X,I0))")') num_layers
    write(unit,trim(fmt)) convolution(:)%stride

    write(fmt,'("(3X,""BIAS ="",",I0,"(1X,F0.9))")') num_layers
    write(unit,trim(fmt)) convolution(:)%bias

    write(unit,'("WEIGHTS")')
    do l=1,num_layers
       write(unit,*) convolution(l)%weight
    end do
    write(unit,'("END WEIGHTS")')
    write(unit,'("END CONVOLUTION")')

    close(unit)

  end subroutine write_file
!!!#############################################################################


!!!#############################################################################
!!! forward propagation pass
!!!#############################################################################
  subroutine forward(input, output)
    implicit none
    real(real12), dimension(padding_lw:,padding_lw:,:), intent(in) :: input
    real(real12), dimension(:,:,:), intent(out) :: output

    integer :: input_channels, num_layers
    integer :: output_size
    integer :: i, j, l, m, ichannel, istride, jstride
    integer :: start_idx, end_idx

    !! get size of the input and output feature maps
    num_layers = size(convolution, dim=1)
    input_channels = size(input, 3)
    output_size = size(output, 1)


    !! Perform the convolution operation
    ichannel = 0
    output = 0._real12
    do l=1,num_layers
       start_idx = -convolution(l)%pad
       end_idx   = convolution(l)%pad + (convolution(l)%centre_width - 1)
       do m=1,input_channels
          ichannel = ichannel + 1

          !! end_stride is the same as output_size
          !! ... hence, forward does not need the fix
          do j=1,output_size
             jstride = (j-1)*convolution(l)%stride + 1
             do i=1,output_size
                istride = (i-1)*convolution(l)%stride + 1

                output(i,j,ichannel) = convolution(l)%bias + &
                     sum( &                
                     input(&
                     istride+start_idx:istride+end_idx,&
                     jstride+start_idx:jstride+end_idx,m) * &
                     convolution(l)%weight &
                )

                output(i,j,ichannel) = transfer%activate(output(i,j,ichannel))

             end do
          end do

       end do
    end do

  end subroutine forward
!!!#############################################################################


!!!#############################################################################
!!! backward propagation pass
!!!#############################################################################
  subroutine backward(input, output_gradients, input_gradients, clip)
    implicit none
    real(real12), dimension(padding_lw:,padding_lw:,:), intent(in) :: input
    real(real12), dimension(:,:,:), intent(in) :: output_gradients
    type(gradient_type), dimension(:), intent(inout) :: input_gradients
    type(clip_type), optional, intent(in) :: clip

    integer :: input_channels, ichannel, num_layers
    integer :: input_lbound, input_ubound
    integer :: l, m, i, j, x, y
    integer :: istride, jstride
    integer :: start_idx, end_idx, output_size, up_idx
    integer :: i_start, i_end, j_start, j_end
    integer :: x_start, x_end, y_start, y_end


    !! get size of the input and output feature maps
    num_layers = size(convolution, dim=1)
    input_channels = size(input, dim=3)
    output_size = size(output_gradients, dim=1)
    input_lbound = lbound(input, dim=1)
    input_ubound = ubound(input, dim=1)

    !! Initialise input_gradients to zero
    do l=1,num_layers
       input_gradients(l)%delta = 0._real12
       input_gradients(l)%weight = 0._real12
       input_gradients(l)%bias = 0._real12
    end do

    !! Perform the convolution operation
    ichannel = 0
    do l=1,num_layers
       start_idx = -convolution(l)%pad
       end_idx   = convolution(l)%pad + (convolution(l)%centre_width - 1)
       up_idx = input_ubound - convolution(l)%kernel_size + 1 - start_idx

       do m=1,input_channels
          ichannel = ichannel + 1

          !! https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/
          !!https://www.youtube.com/watch?v=pUCCd2-17vI

          !! apply full convolution to compute input gradients
          i_input_loop: do j=1,output_size
             j_start = max(1,j+start_idx)
             j_end   = min(output_size,j+end_idx)
             y_start = max(1-j,-end_idx)!max(1-j,start_idx)
             y_end   = min(output_size-j,-start_idx)!min(output_size-j,end_idx)
             jstride = (j-1)*convolution(l)%stride + 1
             j_input_loop: do i=1,output_size
                i_start = max(1,i+start_idx)
                i_end   = min(output_size,i+end_idx)
                x_start = max(1-i,-end_idx)!max(1-i,start_idx)
                x_end   = min(output_size-i,-start_idx)!min(output_size-i,end_idx)
                istride = (i-1)*convolution(l)%stride + 1

                input_gradients(l)%delta(istride,jstride) = &
                     input_gradients(l)%delta(istride,jstride) + &
                     sum( &
                     output_gradients(&
                     i_start:i_end,&
                     j_start:j_end,ichannel) * &
                     convolution(l)%weight(x_start:x_end,y_start:y_end) &
                     ) * &
                     transfer%differentiate(input(istride,jstride,m))
                
             end do j_input_loop
          end do i_input_loop

          !! apply convolution to compute weight gradients
          y_weight_loop: do y=start_idx,end_idx,1
             x_weight_loop: do x=start_idx,end_idx,1
                input_gradients(l)%weight(x,y) = input_gradients(l)%weight(x,y) + &
                     sum(output_gradients(:,:,ichannel) * &
                     input(&
                     x+1:up_idx+x:convolution(l)%stride,&
                     y+1:up_idx+y:convolution(l)%stride,m))
             end do x_weight_loop
          end do y_weight_loop
       !! compute gradients for bias
       !! https://stackoverflow.com/questions/58036461/how-do-you-calculate-the-gradient-of-bias-in-a-conolutional-neural-networo
          input_gradients(l)%bias = input_gradients(l)%bias + &
               sum(output_gradients(:,:,ichannel))
       end do
    end do


    !! apply gradient clipping
    if(present(clip))then
       if(clip%l_min_max) call gradient_clip(input_gradients,&
            clip_min=clip%min,clip_max=clip%max)
       if(clip%l_norm) call gradient_clip(input_gradients,&
            clip_norm=clip%norm)
    end if
    

  end subroutine backward
!!!#############################################################################
  

!!!#############################################################################
!!! update weights and biases according to gradient
!!!#############################################################################
  subroutine update_weights_and_biases(learning_rate, gradients, iteration)
    implicit none
    real(real12), intent(in) :: learning_rate
    type(gradient_type), dimension(:), intent(inout) :: gradients
    integer, optional, intent(inout) :: iteration

    integer :: l,x,y
    integer :: num_layers
    integer :: start_idx, end_idx
    real(real12) :: rtmp1, rtmp2


    !! initialise constants
    num_layers = size(convolution, dim=1)

    !! check if gradients total NaN
    do l=1,num_layers
       if(isnan(sum(gradients(l)%weight)).or.isnan(gradients(l)%bias))then
          write(*,*) gradients(l)%weight
          write(*,*) gradients(l)%bias
          write(0,*) "gradients nan in CV"
          stop
       end if
    end do

    !! update the convolution layer weights using gradient descent
    do l=1,num_layers
       start_idx = -convolution(l)%pad
       end_idx   = convolution(l)%pad + (convolution(l)%centre_width - 1)
       
       !! update the convolution layer weights using gradient descent
       do y=start_idx,end_idx,1
          do x=start_idx,end_idx,1
             if(adaptive_parameters%method.eq.'none')then
                convolution(l)%weight(x,y) = &
                     convolution(l)%weight(x,y) - &
                     learning_rate * gradients(l)%weight(x,y)
             elseif(adaptive_parameters%method.eq.'adam')then
                call update_weight(learning_rate,&
                     convolution(l)%weight(x,y),&
                     convolution(l)%weight_incr(x,y), &
                     gradients(l)%weight(x,y), &
                     gradients(l)%m(x,y), &
                     gradients(l)%v(x,y), &
                     iteration, &
                     adaptive_parameters)
             else
                call update_weight(learning_rate,&
                     convolution(l)%weight(x,y),&
                     convolution(l)%weight_incr(x,y), &
                     gradients(l)%weight(x,y), &
                     rtmp1, &
                     rtmp2, &
                     iteration, &
                     adaptive_parameters)
             end if
          end do
       end do

       !! update the convolution layer bias using gradient descent       
       if(adaptive_parameters%method.eq.'none')then
          convolution(l)%bias = &
               convolution(l)%bias - &
               learning_rate * gradients(l)%bias
       elseif(adaptive_parameters%method.eq.'adam')then
          call update_weight(learning_rate,&
               convolution(l)%bias,&
               convolution(l)%bias_incr, &
               gradients(l)%bias, &
               gradients(l)%bias_m, &
               gradients(l)%bias_v, &
               iteration, &
               adaptive_parameters)
       else
          call update_weight(learning_rate,&
               convolution(l)%bias,&
               convolution(l)%bias_incr, &
               gradients(l)%bias, &
               rtmp1, &
               rtmp2, &
               iteration, &
               adaptive_parameters)
       end if
       
       !! check for NaNs or infinite in weights
       if(any(isnan(convolution(l)%weight)).or.any(convolution(l)%weight.gt.huge(1.E0)))then
          write(0,*) "ERROR: weights in ConvolutionLayer has encountered NaN"
          write(0,*) "Layer:",l
          write(0,*) convolution(l)%weight
          write(0,*) "Exiting..."
          stop
       end if


       !! check for NaNs or infinite in bias
       if(isnan(convolution(l)%bias).or.convolution(l)%bias.gt.huge(1.E0))then
          write(0,*) "ERROR: biases in ConvolutionLayer has encountered NaN"
          write(0,*) "Exiting..."
          stop
       end if

    end do

  end subroutine update_weights_and_biases
!!!#############################################################################


!!!#############################################################################
!!! determine start and end indices for non-full convolution
!!!#############################################################################
  subroutine get_stride_start_end(start_idx,end_idx,width,kernel_size,idx)
    implicit none
    integer, intent(inout) :: start_idx, end_idx
    integer, intent(in) :: width, kernel_size, idx

    if(idx.lt.1)then
       start_idx = 1-idx
    else
       start_idx = 1
    end if

    if(idx.gt.width)then
       end_idx = kernel_size + &
            width-idx
    else
       end_idx = kernel_size
    end if

  end subroutine get_stride_start_end
!!!#############################################################################


!!!#############################################################################
!!! gradient clipping
!!!#############################################################################
  subroutine gradient_clip(gradients, clip_min, clip_max, clip_norm)
    implicit none
    type(gradient_type), dimension(:), intent(inout) :: gradients
    real(real12), optional, intent(in) :: clip_min, clip_max, clip_norm

    integer :: i,j,l, num_layers
    real(real12) :: norm

    num_layers = size(convolution, dim=1)
    if(present(clip_norm))then
       do l=1,num_layers
          norm = sqrt(sum(gradients(l)%weight(:,:)**2._real12) + &
               gradients(l)%bias**2._real12)
          if(norm.gt.clip_norm)then
             gradients(l)%weight(:,:) = &
                  gradients(l)%weight(:,:) * clip_norm/norm
             gradients(l)%bias = &
                  gradients(l)%bias * clip_norm/norm
          end if
       end do
    elseif(present(clip_min).and.present(clip_max))then
       do l=1,num_layers
          do j=lbound(gradients(l)%weight,dim=2),ubound(gradients(l)%weight,dim=2)
             do i=lbound(gradients(l)%weight,dim=1),ubound(gradients(l)%weight,dim=1)
                gradients(l)%weight(i,j) = &
                  max(clip_min,min(clip_max,gradients(l)%weight(i,j)))
             end do
          end do
          gradients(l)%bias = max(clip_min,min(clip_max,gradients(l)%bias))
       end do
    end if

  end subroutine gradient_clip
!!!#############################################################################


end module ConvolutionLayer
!!!#############################################################################
