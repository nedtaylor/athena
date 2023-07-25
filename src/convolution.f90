!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module ConvolutionLayer
  use constants, only: real12
  use custom_types, only: clip_type, convolution_type, activation_type, &
       initialiser_type, learning_parameters_type
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

    output%bias  = a%bias + b%bias
    if(allocated(a%weight).and.allocated(b%weight))then
       allocate(output%weight,mold=a%weight)
       output%weight = a%weight + b%weight
       if(allocated(a%m))then
          output%m = a%m
          output%bias_m = a%bias_m
       end if
       if(allocated(a%v))then
          output%v = a%v
          output%bias_v = a%bias_v
       end if
    end if
    if(allocated(a%delta).and.allocated(b%delta))then
       allocate(output%delta,mold=a%delta)
       output%delta = a%delta + b%delta
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
    use random, only: random_setup
    use misc_ml, only: get_padding_half
    use activation,  only: activation_setup
    use initialiser, only: initialiser_setup
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


    !!-----------------------------------------------------------------------
    !! set learning parameters
    !!-----------------------------------------------------------------------
    if(present(learning_parameters))then
       adaptive_parameters = learning_parameters
    else
       adaptive_parameters%method = "none"
    end if


    !! if file, read in weights and biases
    !! ... if no file is given, weights and biases to a default
    if(present(file))then
       !!-----------------------------------------------------------------------
       !! read convolution layer data from file
       !!-----------------------------------------------------------------------
       write(*,*) "Reading convolutional layers from "//trim(file)
       call read_file(file)
    elseif(present(num_layers).and.present(kernel_size).and.present(stride))then
       
       !!-----------------------------------------------------------------------
       !! set defaults if not present
       !!-----------------------------------------------------------------------
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

       !!-----------------------------------------------------------------------
       !! initialise random seed
       !!-----------------------------------------------------------------------
       if(present(seed))then
          call random_setup(seed, num_seed=1, restart=.false.)
       else
          call random_setup(num_seed=1, restart=.false.)
       end if

       !!-----------------------------------------------------------------------
       !! determine initialisation method
       !!-----------------------------------------------------------------------
       kernel_init = initialiser_setup(t_kernel_initialiser)
       bias_init = initialiser_setup(t_bias_initialiser)

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
          !!--------------------------------------------------------------------
          if(t_full_padding)then
             convolution(l)%pad  = itmp1 - 1
          else
             convolution(l)%pad = get_padding_half(itmp1)
          end if
       
          !! odd or even kernel/filter size
          !!--------------------------------------------------------------------
          convolution(l)%centre_width = 2 - mod(itmp1,2)
       
          !! initialise delta_weight
          !! ... (weight_incr, store of previous weight incremenet)
          !!--------------------------------------------------------------------
          start_idx = -convolution(l)%pad
          end_idx   = convolution(l)%pad + (convolution(l)%centre_width - 1)
          allocate(convolution(l)%weight_incr(start_idx:end_idx,start_idx:end_idx))
          convolution(l)%weight_incr(:,:) = 0._real12

          !! initialise weights and biases according to defined method
          !!--------------------------------------------------------------------
          allocate(convolution(l)%weight(start_idx:end_idx,start_idx:end_idx))
          call kernel_init%initialise(convolution(l)%weight, itmp1*itmp1+1, 1)
          call bias_init%initialise(convolution(l)%bias, itmp1*itmp1+1, 1)

       end do

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
       write(*,'("CV activation function: ",A)') trim(t_activation_function)
       transfer = activation_setup(t_activation_function, scale)

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
    use misc, only: to_lower, icount
    use infile_tools, only: assign_val, assign_vec
    use misc_ml, only: get_padding_half
    use activation,  only: activation_setup
    implicit none
    character(*), intent(in) :: file

    integer :: i, j, k, l, c, istart, istart_weights, itmp1
    integer :: start_idx, end_idx, num_filters, num_inputs
    integer :: unit, stat
    real(real12) :: activation_scale
    character(20) :: activation_function
    character(20) :: padding_type
    character(6) :: line_no
    character(1024) :: buffer, tag
    logical :: found, found_num_filters
    integer, allocatable, dimension(:) :: itmp_list
    real(real12), allocatable, dimension(:) :: data_list


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
       istart_weights = 0
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
          if(trim(adjustl(buffer)).eq."") cycle tag_loop

          !! check for end of convolution card
          if(trim(adjustl(buffer)).eq."END CONVOLUTION")then
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
                allocate(itmp_list(num_filters))
                rewind(unit)
                do j=1,istart
                   read(unit,*)
                end do
                i = istart
             end if
             cycle tag_loop
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
             call assign_vec(buffer, itmp_list, itmp1)
             convolution(:)%kernel_size = itmp_list
          case("STRIDE")
             call assign_vec(buffer, itmp_list, itmp1)
             convolution(:)%stride = itmp_list
          !case("BIAS")
          !   call assign_vec(buffer, convolution(:)%bias, itmp1)
          case("WEIGHTS")
             istart_weights = i
             cycle tag_loop
          case default
             !! don't look for "e" due to scientific notation of numbers
             !! ... i.e. exponent (E+00)
             if(scan(to_lower(trim(adjustl(buffer))),&
                  'abcdfghijklmnopqrstuvwxyz').eq.0)then
                cycle tag_loop
             elseif(tag(:3).eq.'END')then
                cycle tag_loop
             end if
             stop "Unrecognised line in cnn input file: "//trim(adjustl(buffer))
          end select
       end do tag_loop

       !! set transfer activation function
       transfer = activation_setup(trim(activation_function), activation_scale)

       !! check if WEIGHTS card was found
       if(istart_weights.le.0)then
          stop "WEIGHTS card in CONVOLUTION not found!"
       end if

       !! rewind file to WEIGHTS tag
       rewind(unit)
       do j=1,istart_weights
          read(unit,*)
       end do

       !! allocate convolutional layer and read weights
       do l=1,num_filters
          !! padding width
          if(trim(padding_type).eq."full")then
             convolution(l)%pad  = convolution(l)%kernel_size - 1
          elseif(trim(padding_type).eq."none".or.padding_type.eq."valid")then
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
          convolution(l)%bias = 0._real12
          convolution(l)%weight = 0._real12

          num_inputs = convolution(l)%kernel_size ** 2 + 1 !+1 for bias
          allocate(data_list(num_inputs))

          c = 1
          k = 1
          data_list = 0._real12
          data_concat_loop: do while(c.le.num_inputs)
             read(unit,'(A)',iostat=stat) buffer
             if(stat.ne.0) exit data_concat_loop
             k = icount(buffer)
             read(buffer,*,iostat=stat) (data_list(j),j=c,c+k-1)
             c = c + k
          end do data_concat_loop
          convolution(l)%weight(:,:) = &
               reshape(&
               data_list(1:num_inputs-1),&
               shape(convolution(l)%weight(:,:)))
          convolution(l)%bias = data_list(num_inputs)
          deallocate(data_list)
       end do

       !! check for end of weights card
       read(unit,'(A)') buffer
       if(trim(adjustl(buffer)).ne."END WEIGHTS")then
          write(line_no,'(I0)') num_filters + istart_weights + 1
          write(*,*) trim(adjustl(buffer))
          stop "ERROR: END WEIGHTS not where expected, line "//trim(line_no)
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
    character(:), allocatable :: padding_type

    padding_type = ""
    if(convolution(1)%pad.eq.convolution(1)%kernel_size-1)then
       padding_type = "full"
    elseif(convolution(1)%pad.eq.0)then
       padding_type = "valid"
    else
       padding_type = "same"
    end if
       

    unit = 10
    
    open(unit, file=trim(file), access='append')

    num_layers = size(convolution,dim=1)
    write(unit,'("CONVOLUTION")')
    write(unit,'(3X,"NUM_FILTERS = ",I0)') size(convolution,dim=1)
    write(unit,'(3X,"PADDING_TYPE = ",A)') padding_type
    write(unit,'(3X,"ACTIVATION_FUNCTION = ",A)') trim(transfer%name)
    write(unit,'(3X,"ACTIVATION_SCALE = ",F0.9)') transfer%scale

    write(fmt,'("(3X,""KERNEL_SIZE ="",",I0,"(1X,I0))")') num_layers
    write(unit,trim(fmt)) convolution(:)%kernel_size

    write(fmt,'("(3X,""STRIDE ="",",I0,"(1X,I0))")') num_layers
    write(unit,trim(fmt)) convolution(:)%stride

    write(fmt,'("(3X,""BIAS ="",",I0,"(1X,F0.9))")') num_layers
    !write(unit,trim(fmt)) convolution(:)%bias

    write(unit,'("WEIGHTS")')
    do l=1,num_layers
       write(unit,'(5(E16.8E2))', advance="no") convolution(l)%weight
       write(unit,'(E16.8E2)') convolution(l)%bias
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

    integer :: input_channels
    integer :: output_size
    integer :: i, j, l, m, ichannel, istride, jstride
    integer :: start_idx, end_idx


    !! get size of the input and output feature maps
    input_channels = size(input, dim=3)
    output_size = size(output, dim=1)

    !! Perform the convolution operation
    ichannel = 0
    do l=1,size(convolution, dim=1)
       start_idx = -convolution(l)%pad
       end_idx   = convolution(l)%pad + (convolution(l)%centre_width - 1)
       do m=1,input_channels
          ichannel = ichannel + 1

          !! end_stride is the same as output_size
          !! ... hence, forward does not need the fix
          do j=1,output_size,1
             jstride = (j-1)*convolution(l)%stride + 1
             do i=1,output_size,1
                istride = (i-1)*convolution(l)%stride + 1
          
                output(i,j,ichannel) = transfer%activate(convolution(l)%bias + &
                     sum( &                
                     input(&
                     istride+start_idx:istride+end_idx,&
                     jstride+start_idx:jstride+end_idx,m) * &
                     convolution(l)%weight &
                ))
          
             end do
          end do

       end do
    end do

  end subroutine forward
!!!#############################################################################


!!!#############################################################################
!!! backward propagation pass
!!!#############################################################################
  subroutine backward(input, output_gradients, input_gradients)
    implicit none
    real(real12), dimension(padding_lw:,padding_lw:,:), intent(in) :: input
    real(real12), dimension(:,:,:), intent(in) :: output_gradients
    type(gradient_type), dimension(:), intent(inout) :: input_gradients

    integer :: input_channels, ichannel
    integer :: input_lbound, input_ubound
    integer :: l, m, i, j, x, y
    integer :: istride, jstride
    integer :: start_idx, end_idx, output_size, up_idx
    integer :: i_start, i_end, j_start, j_end
    integer :: x_start, x_end, y_start, y_end
    real(real12) :: bias_diff


    !! get size of the input and output feature maps
    input_channels = size(input, dim=3)
    output_size = size(output_gradients, dim=1)
    input_lbound = lbound(input, dim=1)
    input_ubound = ubound(input, dim=1)
    bias_diff = transfer%differentiate(1._real12)

    !! Perform the convolution operation
    ichannel = 0
    do l=1,size(convolution, dim=1)
       input_gradients(l)%delta = 0._real12
       input_gradients(l)%weight = 0._real12
       input_gradients(l)%bias = 0._real12
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
          !! https://saturncloud.io/blog/how-to-calculate-the-gradient-of-bias-in-a-convolutional-neural-network/
          input_gradients(l)%bias = input_gradients(l)%bias + &
               sum(output_gradients(:,:,ichannel)) * bias_diff
       
       end do
    end do

  end subroutine backward
!!!#############################################################################
  

!!!#############################################################################
!!! update weights and biases according to gradient
!!!#############################################################################
  subroutine update_weights_and_biases(learning_rate, gradients, clip, iteration)
    use misc_ml, only: update_weight
    implicit none
    real(real12), intent(in) :: learning_rate
    type(gradient_type), dimension(:), intent(inout) :: gradients
    integer, optional, intent(in) :: iteration
    type(clip_type), optional, intent(in) :: clip

    integer :: l


    !! apply gradient clipping
    if(present(clip))then
       if(clip%l_min_max) call gradient_clip(gradients,&
            clip_min=clip%min,clip_max=clip%max)
       if(clip%l_norm) call gradient_clip(gradients,&
            clip_norm=clip%norm)
    end if

    !! update the convolution layer weights using gradient descent
    do l=1,size(convolution, dim=1)
       !! update the convolution layer weights using gradient descent
       select case(allocated(gradients(l)%m))
       case(.true.)
          call update_weight(learning_rate,&
               convolution(l)%weight,&
               convolution(l)%weight_incr, &
               gradients(l)%weight, &
               iteration, &
               adaptive_parameters, &
               gradients(l)%m, &
               gradients(l)%v)
       case default
          call update_weight(learning_rate,&
               convolution(l)%weight,&
               convolution(l)%weight_incr, &
               gradients(l)%weight, &
               iteration, &
               adaptive_parameters)          
       end select
       !! update the convolution layer bias using gradient descent
       call update_weight(learning_rate,&
            convolution(l)%bias,&
            convolution(l)%bias_incr, &
            gradients(l)%bias, &
            iteration, &
            adaptive_parameters, &
            gradients(l)%bias_m, &
            gradients(l)%bias_v)

       !!! check if gradients total NaN
       !if(isnan(sum(gradients(l)%weight)).or.isnan(convolution(l)%bias))then
       !   write(*,*) gradients(l)%weight
       !   write(*,*) gradients(l)%bias
       !   write(0,*) "ERROR: weights or biases in ConvolutionLayer has encountered NaN"
       !   write(0,*) "Layer:",l
       !   write(0,*) convolution(l)%weight, convolution(l)%bias
       !   write(0,*) "Exiting..."
       !   stop
       !end if

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
    real(real12) :: scale

    !! clipping is not applied to deltas
    num_layers = size(convolution, dim=1)
    if(present(clip_norm))then
       do l=1,num_layers
          scale = min(1._real12, &
               clip_norm/sqrt(sum(gradients(l)%weight**2._real12) + &
               gradients(l)%bias**2._real12))
          if(scale.lt.1._real12)then
             gradients(l)%weight = &
                  gradients(l)%weight * scale
             gradients(l)%bias = &
                  gradients(l)%bias * scale
          end if
       end do
    elseif(present(clip_min).and.present(clip_max))then
       do l=1,num_layers
          do j=lbound(gradients(l)%weight,dim=2),ubound(gradients(l)%weight,dim=2),1
             do i=lbound(gradients(l)%weight,dim=1),ubound(gradients(l)%weight,dim=1),1
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
