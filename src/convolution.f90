!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module ConvolutionLayer
  use constants, only: real12
  use nn_types, only: clip_type
  implicit none

  integer, allocatable, dimension(:) :: half

  type index_type
     integer, allocatable, dimension(:) :: idx
  end type index_type
  type(index_type), allocatable, dimension(:) :: idx_list

  type convolution_type
     integer :: kernel_size
     integer :: stride
     real(real12) :: delta
     real(real12) :: bias
     !! DO THE WEIGHTS NEED TO BE DIFFERENT PER INPUT CHANNEL?
     !! IF SO, 3 DIMENSIONS. IF NOT, 2 DIMENSIONS
     real(real12), allocatable, dimension(:,:) :: weight, weight_incr
     !real(real12), allocatable, dimension(:,:,:) :: output
  end type convolution_type
  type(convolution_type), allocatable, dimension(:) :: convolution

!!! NEED TO MAKE A TYPE TO HANDLE THE OUTPUT OF EACH CONVOLUTION LAYER
!!! THIS IS BECAUSE, if we use different stride for ...
!!! ... each layer, then the dimensions of the output arrays will be ...
!!! ... different for each layer

  private

  public :: convolution
  public :: initialise, forward, backward
  public :: update_weights_and_biases
  public :: write_file
  


contains

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
  subroutine initialise(input_size, seed, num_layers, kernel_size, stride, file)
    implicit none
    integer, intent(in) :: input_size
    integer, intent(in), optional :: seed
    integer, intent(in), optional :: num_layers
    integer, dimension(:), intent(in), optional :: kernel_size, stride
    character(*), optional, intent(in) :: file

    integer :: itmp1,itmp2,nseed
    integer, allocatable, dimension(:) :: seed_arr

    integer :: l,i
    

!!!! num_layers has taken over for output_channels (or cv_num_filters)


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
       call random_seed(size=nseed)
       allocate(seed_arr(nseed))
       if(present(seed))then
          seed_arr = seed
       else
          call system_clock(count=itmp1)
          seed_arr = itmp1 + 37* (/ (l-1,l=1,nseed) /)
       end if
       call random_seed(put=seed_arr)

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
          allocate(convolution(l)%weight(itmp1,itmp1))
          call random_number(convolution(l)%bias)
          call random_number(convolution(l)%weight)
          allocate(convolution(l)%weight_incr(itmp1,itmp1))
          convolution(l)%weight_incr(:,:) = 0._real12
       end do
    else
       write(0,*) "ERROR: Not enough optional arguments provided to initialse CV"
       write(0,*) "Either provide (file) or (num_layers, kernel_size, and stride)"
       write(0,*) "... seed is also optional for the latter set)"
       write(0,*) "Exiting..."
       stop
    end if


    !! get stride information
    if(.not.allocated(idx_list).or..not.allocated(half))then
       allocate(half(num_layers))
       allocate(idx_list(num_layers))
       do l=1,num_layers
          half(l) = convolution(l)%kernel_size/2
          allocate(idx_list(l)%idx(1-half(l):input_size+half(l)))
          do i=1-half(l),input_size+half(l),1
             if(i.lt.1)then
                idx_list(l)%idx(i) = input_size + i
             elseif(i.gt.input_size)then
                idx_list(l)%idx(i) = i - input_size
             else
                idx_list(l)%idx(i) = i
             end if
          end do
       end do
    end if
    

 
  end subroutine initialise
!!!#############################################################################


!!!#############################################################################
!!! 
!!!#############################################################################
  subroutine read_file(file)
    implicit none
    character(*), intent(in) :: file

    integer :: i,j,k,l
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
    integer :: unit=10
    character(128) :: fmt

    
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
    real(real12), dimension(:,:,:), intent(in) :: input
    real(real12), dimension(:,:,:), intent(out) :: output

    integer :: input_channels, num_layers
    integer :: input_size, output_size
    integer :: i, j, l, m, x, y, ichannel, istride, jstride

    !! get size of the input and output feature maps
    num_layers = size(convolution, dim=1)
    input_size = size(input, 1)
    input_channels = size(input, 3)
    output_size = size(output, 1)

    !! Perform the convolution operation
    ichannel = 0
    output = 0._real12
    do l=1,num_layers
       do m=1,input_channels
          ichannel = ichannel + 1

          do i=1,output_size
             istride = (i-1)*convolution(l)%stride-half(l)
             do j=1,output_size
                jstride = (j-1)*convolution(l)%stride-half(l)

                output(i,j,ichannel) = convolution(l)%bias

                do x=1,convolution(l)%kernel_size
                   do y=1,convolution(l)%kernel_size
                      output(i,j,ichannel) = output(i,j,ichannel) + &
                           input(&
                           idx_list(l)%idx(istride+x),&
                           idx_list(l)%idx(jstride+y),&
                           m) * convolution(l)%weight(x,y)
                   end do
                end do

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
    real(real12), dimension(:,:,:), intent(in) :: input
    real(real12), dimension(:,:,:), intent(in) :: output_gradients
    real(real12), dimension(:,:,:), intent(out) :: input_gradients
    type(clip_type), optional, intent(in) :: clip

    integer :: input_channels, output_channels, ichannel, num_layers
    integer :: i, j, k, l, m, n, x, y, x180, y180
    integer :: istride, jstride
    real(real12) :: gradient_norm
    integer :: output_size

    !! Initialise input_gradients to zero
    input_gradients = 0._real12

    !! get size of the input and output feature maps
    num_layers = size(convolution, dim=1)
    input_channels = size(input, 3)
    output_size = size(output_gradients, 1)

    !! Perform the convolution operation
    ichannel = 0
    do l=1,num_layers
       do m=1,input_channels
          ichannel = ichannel + 1
          do i=1,output_size
             istride = (i-1)*convolution(l)%stride-half(l)
             do j=1,output_size
                jstride = (j-1)*convolution(l)%stride-half(l)

                do x=1,convolution(l)%kernel_size
                   x180 = convolution(l)%kernel_size - x + 1
                   do y=1,convolution(l)%kernel_size
                      y180 = convolution(l)%kernel_size - y + 1
                      !! Compute gradients for input feature map
                      input_gradients(&
                           idx_list(l)%idx(istride+x),&
                           idx_list(l)%idx(jstride+y), m) = &
                           input_gradients(&
                           idx_list(l)%idx(istride+x),&
                           idx_list(l)%idx(jstride+y), m) + &
                           convolution(l)%weight(x180,y180) * &
                           output_gradients(i,j,ichannel)
                           !convolution(l)%weight(x,y) * &
                           !output_gradients(i,j,ichannel)

!!! TWO OPTIONS?
!!! FIND THE GRADIENT/LOSS OF THE WEIGHTS:
!!! ... dL/dF = CONVOL(input, output_gradients)
!!! ... dL/dX = FULLCONV(180deg rotated weights, output_gradients)
!!! ... where: F = filter/weights, X = input
!!! NOTE:
!!! CONVOL = filter applied to non-padded input (i.e. size(output) < size(input))
!!! FULLCONVOL = filter applied to padded input (i.e. size(output) = size(input))

!!! WARNING!
!!! DOING THE FULL CONVOLUTION THE WAY I HAVE (PERIODICITY) WORKS FOR CHGCARS
!!! HOWEVER, LIKELY NOT GOOD TO USE THIS WITH IMAGES OF WRITTEN TEXT

                   end do
                end do

             end do
          end do
       end do
    end do
    
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
    integer :: i,j,l,m,x,y,istride,jstride
    integer :: input_channels, num_layers, grad_size, ichannel
    real(real12) :: sum_gradients
    real(real12), optional, intent(in) :: l1_lambda, l2_lambda, momentum
    real(real12), intent(in) :: learning_rate
    real(real12), dimension(:,:,:), intent(in) :: input
    real(real12), dimension(:,:,:), intent(in) :: gradients


    !! Check if gradients total NaN
    if(isnan(sum(gradients)))then
       write(0,*) "gradients nan in CV"
       return
    end if

    !! Initialise constants
    num_layers = size(convolution, dim=1)
    input_channels = size(gradients,3)/num_layers
    grad_size = size(gradients,1)
    ichannel = 0
    
    !! Update the convolution layer weights using gradient descent
    do l=1,num_layers
       sum_gradients = 0._real12
       do m=1,input_channels
          ichannel = ichannel + 1
          sum_gradients = sum_gradients + sum(gradients(:,:,ichannel))

          do i=1,grad_size
             istride = (i-1)*convolution(l)%stride-half(l)
             do j=1,grad_size
                jstride = (j-1)*convolution(l)%stride-half(l)
                
                do x=1,convolution(l)%kernel_size
                   !x180 = convolution(l)%kernel_size - x + 1
                   do y=1,convolution(l)%kernel_size
                      !y180 = convolution(l)%kernel_size - y + 1
                     
                      
                      !! momentum-based learning
                      if(present(momentum))then
                         convolution(l)%weight_incr(x,y) = &
                              learning_rate * gradients(&
                              idx_list(l)%idx(istride+x),&
                              idx_list(l)%idx(jstride+y), ichannel) + &
                              momentum * convolution(l)%weight_incr(x,y)
                         !convolution(l)%weight(x,y) = convolution(l)%weight(x,y) - &
                         !     convolution(l)%weight_incr(x,y)
                      else
                         !convolution(l)%weight(x,y) = convolution(l)%weight(x,y) - &
                         !     learning_rate * gradients(&
                         !     idx_list(l)%idx(istride+x),&
                         !     idx_list(l)%idx(jstride+y), ichannel)
                         convolution(l)%weight_incr(x,y) = &
                              learning_rate * gradients(&
                              idx_list(l)%idx(istride+x),&
                              idx_list(l)%idx(jstride+y), ichannel)   
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

             end do
          end do

       end do

       !do i=1,size(weights,4)
       !   weights(:,:,:,i) = weights(:,:,:,i) - learning_rate * gradients
       !end do
       if(any(isnan(convolution(l)%weight)).or.any(convolution(l)%weight.gt.huge(1.E0)))then
          write(0,*) "ERROR: weights in ConvolutionLayer has encountered NaN"
          write(0,*) "Layer:",l
          write(0,*) convolution(l)%weight
          write(0,*) "Exiting..."
          stop
       end if
       
       !! Update the convolution layer biases using gradient descent
       convolution(l)%bias = convolution(l)%bias - &
            learning_rate * sum_gradients
       
       if(isnan(convolution(l)%bias).or.convolution(l)%bias.gt.huge(1.E0))then
          write(0,*) "ERROR: biases in ConvolutionLayer has encountered NaN"
          write(0,*) "Exiting..."
          stop
       end if

    end do

  end subroutine update_weights_and_biases
!!!#############################################################################


!!!#############################################################################
!!! gradient clipping
!!!#############################################################################
  subroutine gradient_clip(gradients, clip_min, clip_max, clip_norm)
    implicit none
    real(real12), dimension(:,:,:) :: gradients
    real(real12), optional, intent(in) :: clip_min, clip_max, clip_norm

    integer :: i,j,k,l, input_channels, num_layers
    real(real12) :: norm

    num_layers = size(convolution, dim=1)
    input_channels = size(gradients,dim=3)/num_layers
    if(present(clip_norm))then
       do l=1,num_layers
          norm = norm2(gradients(:,:,(l-1)*input_channels+1:l*input_channels))
          if(norm.gt.clip_norm)then
             gradients(:,:,(l-1)*input_channels+1:l*input_channels) = &
                  gradients(:,:,(l-1)*input_channels+1:l*input_channels) * clip_norm/norm
          end if
       end do
    elseif(present(clip_min).and.present(clip_max))then
       do k=1,size(gradients,dim=3)
          do j=1,size(gradients,dim=2)
             do i=1,size(gradients,dim=1)
                gradients(i,j,k) = &
                  max(clip_min,min(clip_max,gradients(i,j,k)))
             end do
          end do
       end do
    end if

  end subroutine gradient_clip
!!!#############################################################################


end module ConvolutionLayer
!!!#############################################################################
