!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module ConvolutionLayer
  use constants, only: real12
  use random, only: random_setup
  use custom_types, only: clip_type, convolution_type, activation_type, &
       initialiser_type
  use misc_ml, only: get_padding_half
  use activation,  only: activation_setup
  use initialiser, only: initialiser_setup
  implicit none


  !! https://www.nag.com/nagware/np/r62_doc/manual/compiler_9_2.html
  !! https://www.intel.com/content/www/us/en/docs/fortran-compiler/developer-guide-reference/2023-0/generic.html
  type gradient_type
     real(real12), allocatable, dimension(:,:) :: delta
     real(real12), allocatable, dimension(:,:) :: weight
     real(real12) :: bias
   contains
     procedure :: add_t_t => gradient_add
     generic :: operator(+) => add_t_t !, public
  end type gradient_type

  class(activation_type), allocatable :: transfer!activation


  integer :: padding_lw

  type(convolution_type), allocatable, dimension(:) :: convolution

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
!!!
!!!#############################################################################
  elemental function gradient_add(a,b) result(output)
    implicit none
    class(gradient_type), intent(in) :: a,b
    type(gradient_type) :: output
  
    allocate(output%weight,mold=a%weight)
    output%weight = a%weight + b%weight
    output%bias  = a%bias + b%bias
    output%delta = a%delta + b%delta
        
  end function gradient_add
!!!#############################################################################


!!!#############################################################################
!!!
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
!!!
!!!#############################################################################
  subroutine initialise(seed, num_layers, kernel_size, stride, file, &
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
    stop


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



  end subroutine initialise
!!!#############################################################################


!!!#############################################################################
!!!
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
       end do
    else
       do l=1,num_filters
          gradients(l)%weight = 0._real12
          gradients(l)%bias = 0._real12
          gradients(l)%delta = 0._real12
       end do
    end if

    return
  end subroutine allocate_gradients
!!!#############################################################################


!!!#############################################################################
!!!
!!!#############################################################################
  subroutine initialise_gradients(gradients, input_size)
    implicit none
    integer :: l, start_idx, end_idx
    integer, intent(in) :: input_size
    type(gradient_type), allocatable, dimension(:), intent(out) :: gradients
    
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
    end do

    return
  end subroutine initialise_gradients
!!!#############################################################################


!!!#############################################################################
!!! 
!!!#############################################################################
  subroutine read_file(file)
    implicit none
    character(*), intent(in) :: file

    !integer :: i,j,k,l
    integer :: unit,stat,completed
    character(1024) :: buffer
    logical :: found


    if(len(trim(file)).gt.0)then
       unit = 10
       found = .false.
       open(unit, file=trim(file))
       do while (.not.found)
          read(unit,'(A)',iostat=stat) buffer
          if(stat.ne.0)then
             write(0,*) "ERROR: file hit error (EoF?) before CONV_LAYER section"
             write(0,*) "Exiting..."
             stop
          end if
          if(trim(adjustl(buffer)).eq."CONV_LAYER") found = .true.
       end do

       !read(unit,*) kernel_size, input_channels, output_channels, stride
       read(unit,*)

       completed = 0
       !do while (completed.lt.2)
       !   
       !   read(unit,'(A)',iostat=stat) buffer
       !   if(stat.ne.0)then
       !      write(0,*) "ERROR: file hit error (EoF?) before encountering END CONV_LAYER"
       !      write(0,*) "Exiting..."
       !      stop
       !   end if
       !   i = 0
       !   found = .false.
       !   if(trim(adjustl(buffer)).eq."WEIGHTS")then
       !      do while (.not.found)
       !         read(unit,'(A)',iostat=stat) buffer
       !         if(stat.ne.0)then
       !            write(0,*) "ERROR: file hit error (EoF?) before encountering END"
       !            write(0,*) "Exiting..."
       !            stop
       !         end if
       !         if(index(trim(adjustl(buffer)),"END").ne.1)then
       !            found = .true.
       !            completed = completed + 1
       !            cycle
       !         end if
       !         if(trim(adjustl(buffer)).eq."") cycle
       !
       !         i = i + 1
       !         if(i.gt.kernel_size)then
       !            write(0,*) "ERROR: i exceeded kernel_size in CONV_LAYER"
       !            write(0,*) "Exiting..."
       !            stop
       !         end if
       !         read(buffer,*) (((weights(i,j,k,l),&
       !              l=1,output_channels),&
       !              k=1,input_channels),&
       !              j=1,kernel_size)
       !      end do
       !   elseif(trim(adjustl(buffer)).eq."BIASES")then
       !      do while (.not.found)
       !         read(unit,'(A)',iostat=stat) buffer
       !         if(stat.ne.0)then
       !            write(0,*) "ERROR: file hit error (EoF?) before encountering END"
       !            write(0,*) "Exiting..."
       !            stop
       !         end if
       !         if(index(trim(adjustl(buffer)),"END").ne.1)then
       !            found = .true.
       !            completed = completed + 1
       !            cycle
       !         end if
       !         if(trim(adjustl(buffer)).eq."") cycle
       !
       !         i = i + 1
       !         if(i.gt.kernel_size)then
       !            write(0,*) "ERROR: i exceeded kernel_size in CONV_LAYER"
       !            write(0,*) "Exiting..."
       !            stop
       !         end if
       !         read(buffer,*) (biases(l),l=1,output_channels)
       !      end do
       !   end if
       !end do
       close(unit)

       return
    end if

  end subroutine read_file
!!!#############################################################################


!!!#############################################################################
!!! 
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
!!! 
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
!!! 
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
!!! 
!!!#############################################################################
  subroutine update_weights_and_biases(learning_rate, input, gradients, &
       l1_lambda, l2_lambda, momentum)
    implicit none
    integer :: l,x,y
    integer :: num_layers
    integer :: start_idx, end_idx
    real(real12), optional, intent(in) :: l1_lambda, l2_lambda, momentum
    real(real12), intent(in) :: learning_rate
    real(real12), dimension(padding_lw:,padding_lw:,:), intent(in) :: input
    type(gradient_type), dimension(:), intent(in) :: gradients

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
       
       do y=start_idx,end_idx,1
          do x=start_idx,end_idx,1

             !! momentum-based learning
             if(present(momentum))then
                convolution(l)%weight_incr(x,y) = &
                     learning_rate * &
                     gradients(l)%weight(x, y) + &
                     momentum * convolution(l)%weight_incr(x,y)
             else
                convolution(l)%weight_incr(x,y) = &
                     learning_rate * &
                     gradients(l)%weight(x, y)   
             end if

             !! L1 regularisation
             if(present(l1_lambda))then
                convolution(l)%weight_incr(x,y) = &
                     convolution(l)%weight_incr(x,y) + &
                     learning_rate * l1_lambda * &
                     sign(1._real12,convolution(l)%weight(x,y))
             end if

             !! L2 regularisation
             if(present(l2_lambda))then
                convolution(l)%weight_incr(x,y) = &
                     convolution(l)%weight_incr(x,y) + &
                     learning_rate * l2_lambda * convolution(l)%weight(x,y)
             end if

             convolution(l)%weight(x,y) = convolution(l)%weight(x,y) - &
                  convolution(l)%weight_incr(x,y)


          end do
       end do

       !! update the convolution layer biases using gradient descent
       if(present(momentum))then
          convolution(l)%bias = convolution(l)%bias - ( &
               learning_rate * gradients(l)%bias + &
               momentum * convolution(l)%bias )
       else
          convolution(l)%bias = convolution(l)%bias - &
               learning_rate * gradients(l)%bias
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
