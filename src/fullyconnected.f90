!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module FullyConnectedLayer
  use constants, only: real12
  use random, only: random_setup
  use misc_ml, only: update_weight
  use custom_types, only: network_type, clip_type, activation_type, &
       initialiser_type, learning_parameters_type
  use activation,  only: activation_setup
  use initialiser, only: initialiser_setup
  implicit none

  type hidden_output_type
     real(real12), allocatable, dimension(:) :: val
     !contains
     !  final :: hidden_output_type_destructor
  end type hidden_output_type

  type gradient_type
     real(real12), allocatable, dimension(:) :: delta !error !dE/d(activation)
     real(real12), allocatable, dimension(:,:) :: weight
     real(real12), allocatable, dimension(:,:) :: m
     real(real12), allocatable, dimension(:,:) :: v
   contains
     procedure :: add_t_t => gradient_add  !t = type, r = real, i = int
     generic :: operator(+) => add_t_t !, public
     !final :: gradient_type_destructor
  end type gradient_type

  type(network_type), allocatable, dimension(:) :: network
  type(learning_parameters_type) :: adaptive_parameters

  class(activation_type), allocatable :: transfer!activation


  private

  public :: network
  public :: gradient_type
  public :: allocate_gradients
  public :: initialise_gradients

  public :: hidden_output_type

  public :: initialise, forward, backward
  public :: update_weights_and_biases
  public :: write_file


contains

  !subroutine hidden_output_type_destructor(this)
  !  implicit none
  !  type(hidden_output_type) :: this
  !  if(allocated(this%val)) deallocate(this%val)
  !end subroutine hidden_output_type_destructor
  !
  !subroutine gradient_type_destructor(this)
  !  implicit none
  !  type(gradient_type) :: this
  !  
  !  deallocate(this%delta)
  !  deallocate(this%weight)
  !  if(allocated(this%m)) deallocate(this%m)
  !  if(allocated(this%v)) deallocate(this%v)
  !end subroutine gradient_type_destructor

!!!#############################################################################
!!! custom operation for summing gradient_type
!!!#############################################################################
  elemental function gradient_add(a, b) result(output)
    implicit none
    class(gradient_type), intent(in) :: a,b
    type(gradient_type) :: output
    
    if(allocated(a%weight).and.allocated(b%weight))then
       !allocate(output%weight,mold=a%weight)
       output%weight = a%weight + b%weight
       !!output%delta = output%delta + input%delta
       if(allocated(a%m)) output%m = a%m !+ input%m
       if(allocated(a%v)) output%v = a%v !+ input%v
    end if
    if(allocated(a%delta).and.allocated(b%delta))then
       output%delta = a%delta + b%delta
    end if

  end function gradient_add
!!!#############################################################################


!!!#############################################################################
!!!
!!!#############################################################################
  subroutine initialise(seed, num_layers, num_inputs, num_hidden, &
       activation_function, activation_scale, learning_parameters, file, &
       weight_initialiser)
    implicit none
    integer, optional, intent(in) :: seed
    integer, optional, intent(in):: num_layers, num_inputs
    integer, dimension(:), optional, intent(in) :: num_hidden
    real(real12), optional, intent(in) :: activation_scale
    character(*), optional, intent(in) :: file, activation_function, &
         weight_initialiser
    type(learning_parameters_type), optional, intent(in) :: learning_parameters

    integer :: itmp1,nseed, length
    real(real12) :: scale
    character(len=10) :: t_activation_function
    integer, allocatable, dimension(:) :: seed_arr
    class(initialiser_type), allocatable :: weight_init
    character(:), allocatable :: t_weight_initialiser

    integer :: l,i


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
       write(*,*) "Reading fully connected layers from "//trim(file)
       call read_file(file)
    elseif(present(num_layers).and.present(num_inputs).and.&
         present(num_hidden))then
       !!-----------------------------------------------------------------------
       !! initialise random seed
       !!-----------------------------------------------------------------------
       if(present(seed))then
          call random_setup(seed, num_seed=1, restart=.false.)
       else
          call random_setup(num_seed=1, restart=.false.)
       end if

       !!-----------------------------------------------------------------------
       !! set weight initialiser if not present
       !!-----------------------------------------------------------------------
       if(present(weight_initialiser))then
          t_weight_initialiser = weight_initialiser
       else
          t_weight_initialiser = "he_uniform"
       end if
       write(*,'("FC weight initialiser: ",A)') t_weight_initialiser

       !!-----------------------------------------------------------------------
       !! determine initialisation method
       !!-----------------------------------------------------------------------
       weight_init = initialiser_setup(t_weight_initialiser)

       !!-----------------------------------------------------------------------
       !! randomly initialise convolution layers
       !!-----------------------------------------------------------------------
       allocate(network(num_layers))
       do l=1,num_layers
          allocate(network(l)%neuron(num_hidden(l)))
          if(l.eq.1)then
             length = num_inputs+1
          else
             length = num_hidden(l-1)+1
          end if
          
          do i=1,num_hidden(l)
             allocate(network(l)%neuron(i)%weight_incr(length))
             network(l)%neuron(i)%weight_incr = 0._real12

             !! initialise weights according to defined method
             !!-----------------------------------------------------------------
             allocate(network(l)%neuron(i)%weight(length))
             call weight_init%initialise(network(l)%neuron(i)%weight, &
                  fan_in = size(network(l)%neuron(i)%weight,dim=1), &
                  fan_out = num_hidden(l))
          end do

       end do

       !!-----------------------------------------------------------------------
       !! set activation and derivative functions based on input name
       !!-----------------------------------------------------------------------
       if(present(activation_function))then
          t_activation_function = activation_function
       else
          t_activation_function = "relu"
       end if
       if(present(activation_scale))then
          scale = activation_scale
       else
          scale = 1._real12
       end if
       write(*,'("FC activation function: ",A)') trim(t_activation_function)
       transfer = activation_setup(t_activation_function, scale)

    else
       write(0,*) "ERROR: Not enough optional arguments provided to initialse FC"
       write(0,*) "Either provide (file) or (num_layers, num_inputs, and num_hidden)"
       write(0,*) "... seed is also optional for the latter set)"
       write(0,*) "Exiting..."
       stop
    end if


  end subroutine initialise                              
!!!#############################################################################


!!!#############################################################################
!!! 
!!!#############################################################################
  subroutine allocate_gradients(gradients, mold, reallocate)
    implicit none
    type(gradient_type), allocatable, dimension(:), intent(out) :: gradients
    type(gradient_type), dimension(0:), intent(in) :: mold
    logical, optional, intent(in) :: reallocate
    integer :: l
    integer :: num_neurons, num_inputs, num_layers, num_features
    logical :: t_reallocate
    

    if(present(reallocate))then
       t_reallocate = reallocate
    else
       t_reallocate = .true.
    end if
    
    num_layers = ubound(mold,dim=1)
    num_inputs = size(mold(0)%delta,dim=1)
    if(.not.allocated(gradients).or.t_reallocate)then
       if(allocated(gradients)) deallocate(gradients)
       allocate(gradients(0:num_layers))
       allocate(gradients(0)%delta(num_inputs))
       gradients(0)%delta = 0._real12

       do l=1,num_layers,1
          num_neurons = size(mold(l)%delta,dim=1)
          num_inputs = size(mold(l)%weight,dim=1)
          if(allocated(gradients(l)%weight)) deallocate(gradients(l)%weight)
          if(allocated(gradients(l)%delta)) deallocate(gradients(l)%delta)
          allocate(gradients(l)%weight(num_inputs, num_neurons))
          allocate(gradients(l)%delta(num_neurons))
          gradients(l)%weight = 0._real12
          gradients(l)%delta  = 0._real12

          if(allocated(mold(l)%m))then
             if(allocated(gradients(l)%m)) deallocate(gradients(l)%m)
             if(allocated(gradients(l)%v)) deallocate(gradients(l)%v)
             allocate(gradients(l)%m(num_inputs, num_neurons))
             allocate(gradients(l)%v(num_inputs, num_neurons))
             gradients(l)%m = 0._real12
             gradients(l)%v = 0._real12
          end if
       end do
    else
       gradients(0)%delta = 0._real12
       do l=1,num_layers,1
          gradients(l)%weight = 0._real12
          gradients(l)%delta  = 0._real12
          if(allocated(gradients(l)%m))then
             gradients(l)%m = 0._real12
             gradients(l)%v = 0._real12
          end if
       end do
    end if

    return
  end subroutine allocate_gradients
!!!#############################################################################


!!!#############################################################################
!!! 
!!!#############################################################################
  subroutine initialise_gradients(gradients, num_features, adam_learning)
    implicit none
    integer, intent(in) :: num_features
    type(gradient_type), allocatable, dimension(:), intent(out) :: gradients
    logical, optional, intent(in) :: adam_learning

    integer :: l
    integer :: num_neurons, num_inputs, num_layers

    
    num_layers = size(network,dim=1)
    if(allocated(gradients)) deallocate(gradients)
    allocate(gradients(0:num_layers))
    allocate(gradients(0)%delta(num_features))
    gradients(0)%delta = 0._real12
    
    do l=1,num_layers
       num_neurons = size(network(l)%neuron,dim=1)
       num_inputs = size(network(l)%neuron(1)%weight,dim=1)
       if(allocated(gradients(l)%weight)) deallocate(gradients(l)%weight)
       if(allocated(gradients(l)%delta)) deallocate(gradients(l)%delta)
       allocate(gradients(l)%weight(num_inputs, num_neurons))
       allocate(gradients(l)%delta(num_neurons))
       gradients(l)%weight = 0._real12
       gradients(l)%delta  = 0._real12
    
       if(present(adam_learning))then
          if(adam_learning)then
             if(allocated(gradients(l)%m)) deallocate(gradients(l)%m)
             if(allocated(gradients(l)%v)) deallocate(gradients(l)%v)
             allocate(gradients(l)%m(num_inputs, num_neurons))
             allocate(gradients(l)%v(num_inputs, num_neurons))
             gradients(l)%m = 0._real12
             gradients(l)%v = 0._real12
          end if
       end if
    end do
    
    return
  end subroutine initialise_gradients
!!!#############################################################################


!!!#############################################################################
!!! 
!!!#############################################################################
  subroutine read_file(file)
    use misc, only: to_lower, icount
    use infile_tools, only: assign_val, assign_vec
    implicit none
    character(*), intent(in) :: file

    integer :: i, j, k, l, c, istart, istart_weights, itmp1
    integer :: num_layers, num_inputs
    integer :: unit, stat
    real(real12) :: activation_scale
    character(20) :: activation_function
    character(6) :: line_no
    character(1024) :: buffer, tag
    logical :: found, found_num_layers
    integer, allocatable, dimension(:) :: num_hidden
    real(real12), allocatable, dimension(:) :: data_list


    found_num_layers = .false.
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
             write(0,*) "ERROR: file hit error (EoF?) before FULLYCONNECTED section"
             write(0,*) "Exiting..."
             stop
          end if
          if(trim(adjustl(buffer)).eq."FULLYCONNECTED")then
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
             write(0,*) "ERROR: file hit error (EoF?) before encountering END FULLYCONNECTED"
             write(0,*) "Exiting..."
             stop
          end if
          found = .false.
          if(trim(adjustl(buffer)).eq."") cycle tag_loop

          !! check for end of card
          if(trim(adjustl(buffer)).eq."END FULLYCONNECTED")then
             exit tag_loop
          end if
          
          tag=trim(adjustl(buffer))
          if(scan(buffer,"=").ne.0) tag=trim(tag(:scan(tag,"=")-1))

          !! read number of filters from save file
          select case(trim(tag))
          case("NUM_LAYERS")
             if(.not.found_num_layers)then
                call assign_val(buffer, num_layers, itmp1)
                found_num_layers = .true.
                allocate(network(num_layers))
                allocate(num_hidden(num_layers))
                rewind(unit)
                do j=1,istart
                   read(unit,*)
                end do
                i = istart
             end if
             cycle tag_loop
          case default
             if(.not.found_num_layers) cycle tag_loop
          end select

          !! read parameters from save file
          select case(trim(tag))
          case("ACTIVATION_FUNCTION")
             call assign_val(buffer, activation_function, itmp1)
          case("ACTIVATION_SCALE")
             call assign_val(buffer, activation_scale, itmp1)
          case("NUM_HIDDEN")
             call assign_vec(buffer, num_hidden, itmp1)
          case("NUM_INPUTS")
             call assign_val(buffer, num_inputs, itmp1)
          case("WEIGHTS")
             istart_weights = i
             cycle tag_loop
          case default
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
       transfer = activation_setup(activation_function, activation_scale)

       !! check if WEIGHTS card was found
       if(istart_weights.le.0)then
          stop "WEIGHTS card in FULLYCONNECTED not found!"
       end if

       !! rewind file to WEIGHTS tag
       rewind(unit)
       do j=1,istart_weights
          read(unit,*)
       end do
       
       !! allocate layer and read weights and biases
       num_inputs = num_inputs + 1 ! include bias in inputs
       do l=1,num_layers
          allocate(network(l)%neuron(num_hidden(l)))
          do i=1,num_hidden(l)
             allocate(network(l)%neuron(i)%weight(num_inputs))
             allocate(network(l)%neuron(i)%weight_incr(num_inputs))
             network(l)%neuron(i)%weight_incr = 0._real12
             network(l)%neuron(i)%weight = 0._real12

             allocate(data_list(num_inputs))
             data_list = 0._real12
             c = 1
             k = 1
             data_concat_loop: do while(c.le.num_inputs)
                read(unit,'(A)',iostat=stat) buffer
                if(stat.ne.0) exit data_concat_loop
                k = icount(buffer)
                read(buffer,*,iostat=stat) (data_list(j),j=c,c+k-1)
                c = c + k
             end do data_concat_loop
             network(l)%neuron(i)%weight = data_list
             deallocate(data_list)

          end do
          num_inputs = num_hidden(l) + 1
       end do

       !! check for end of weights card
       read(unit,'(A)') buffer
       if(trim(adjustl(buffer)).ne."END WEIGHTS")then
          write(line_no,'(I0)') num_layers + istart_weights + 1
          write(*,*) trim(adjustl(buffer))
          stop "ERROR: END WEIGHTS not where expected, line "//trim(line_no)
       end if
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

    integer :: num_layers,num_inputs,num_weights
    integer :: l,i
    integer :: unit
    character(128) :: fmt
    integer, allocatable, dimension(:) :: num_hidden
    real(real12), allocatable, dimension(:) :: biases

    unit = 10
    open(unit, file=trim(file), access='append')

    num_layers = size(network,dim=1)
    num_inputs = size(network(1)%neuron(1)%weight,dim=1)-1

    allocate(num_hidden(num_layers))
    do l=1,num_layers
       num_hidden(l) = size(network(l)%neuron,dim=1)
    end do
    write(unit,'("FULLYCONNECTED")')
    write(unit,'(3X,"ACTIVATION_FUNCTION = ",A)') trim(transfer%name)
    write(unit,'(3X,"ACTIVATION_SCALE = ",F0.9)') transfer%scale
    write(unit,'(3X,"NUM_INPUTS = ",I0)') num_inputs
    write(unit,'(3X,"NUM_LAYERS = ",I0)') num_layers
    write(fmt,'("(3X,""NUM_HIDDEN ="",",I0,"(1X,I0))")') num_layers
    write(unit,trim(fmt)) num_hidden(:)

    write(unit,'("WEIGHTS")')
    do l=1,num_layers
       num_weights = size(network(l)%neuron(1)%weight,dim=1)
       do i=1,num_hidden(l)
          write(unit,'(5(1X,E15.8E2))') network(l)%neuron(i)%weight(:num_weights)
       end do
    end do
    write(unit,'("END WEIGHTS")')
    write(unit,'("END FULLYCONNECTED")')

    close(unit)
  end subroutine write_file
!!!#############################################################################


!!!#############################################################################
!!! 
!!!#############################################################################
  subroutine forward(input, output)
    implicit none
    real(real12), dimension(:), intent(in) :: input
    type(hidden_output_type), allocatable, dimension(:), intent(out) :: output

    integer :: j, l
    integer :: num_layers, num_neurons
    real(real12) :: activation
    real(real12), allocatable, dimension(:) :: new_input

    !! initialise ouput (and temporary input) arrays
    allocate(new_input(size(input)))
    new_input = input
    num_layers = size(network,dim=1)
    if(allocated(output)) deallocate(output)
    allocate(output(num_layers))

    !! generate outputs from weights, biases, and inputs
    do l=1,num_layers
       num_neurons=size(network(l)%neuron)
       allocate(output(l)%val(num_neurons))
       do j=1,num_neurons
          activation = activate(network(l)%neuron(j)%weight,new_input)
          output(l)%val(j) = transfer%activate(activation)
       end do

       deallocate(new_input)
       if(l.lt.num_layers)then
          allocate(new_input(num_neurons))
          new_input = output(l)%val(:)
       end if
    end do

  end subroutine forward
!!!#############################################################################


!!!#############################################################################
!!! backward propagation
!!! method : gradient descent
!!!#############################################################################
!!! https://brilliant.org/wiki/backpropagation/
  subroutine backward(input, output, expected, input_gradients)
    implicit none
    real(real12), dimension(:), intent(in) :: input
    type(hidden_output_type), allocatable, dimension(:), intent(in) :: output
    real(real12), dimension(:), intent(in) :: expected !is this just output_gradients?
    type(gradient_type), dimension(0:), intent(inout) :: input_gradients
    
    integer :: j, k, l
    integer :: num_layers
    integer :: num_neurons
    real(real12), allocatable, dimension(:) :: new_input
    !type(error_type), dimension(size(network,dim=1)) :: delta !!error

    !!! Initialise input_gradients to zero
    input_gradients(0)%delta = 0._real12
    num_layers = size(network,dim=1)
    do l=1,num_layers
       input_gradients(l)%weight = 0._real12
       input_gradients(l)%delta = 0._real12
       !! if adams optimiser, then initialise these so as not to interfere ...
       !! ... with the combined one
       !if(allocated(input_gradients(l)%m)) input_gradients(l)%m = 0._real12
       !if(allocated(input_gradients(l)%v)) input_gradients(l)%v = 0._real12
    end do


    !! loop through the layers in reverse
    do l=num_layers,0,-1
       if(l.eq.0)then
          num_neurons = size(input,dim=1)
       else
          num_neurons = size(network(l)%neuron,dim=1)
       end if

       !! for first layer, the error is the difference between ...
       !! ... predicted and expected
       if (l.eq.num_layers)then
          input_gradients(l)%delta(:) = expected
       else
          do j=1,num_neurons
             !! the errors are summed from the delta of the ...
             !! ... 'child' node * 'child' weight
             do k=1,size(network(l+1)%neuron,dim=1)
                input_gradients(l)%delta(j) = input_gradients(l)%delta(j) + &
                     ( network(l+1)%neuron(k)%weight(j) * &
                     input_gradients(l+1)%delta(k) )
             end do
          end do
       end if
       !! the delta values are the error multipled by the derivative ...
       !! ... of the transfer function
       !! final layer: error (delta) = activ_diff (g') * error
       !! other layer: error (delta) = activ_diff (g') * sum(weight(l+1)*error(l+1))
       if (l.eq.0)then
          do j=1,num_neurons
             !! here, the activation of the neuron is the input value
             !! ... as each neuron is only connected to one input value
             input_gradients(l)%delta(j) = input_gradients(l)%delta(j) * &
                  transfer%differentiate(input(j))
          end do
       else
          !! define the input to the neuron
          if(l.eq.1)then
             allocate(new_input(size(input,1)))
             new_input = input
          else
             allocate(new_input(size(network(l-1)%neuron(:),1)))
             new_input = output(l-1)%val
          end if
          do j=1,num_neurons
             !! activation already calculated and equals the output
             input_gradients(l)%delta(j) = input_gradients(l)%delta(j) * &
                  transfer%differentiate(output(l)%val(j))
             do k=1,size(new_input,1)
                input_gradients(l)%weight(k,j) = input_gradients(l)%delta(j) * new_input(k)
             end do
          end do
          !! bias weight gradient
          !! ... as the bias neuron = 1._real12, then gradient of the bias ...
          !! ... is just the delta (error), no need to multiply by 1._real12
          input_gradients(l)%weight(size(new_input,1)+1,:) = &
               input_gradients(l)%delta(:)
          deallocate(new_input)
       end if

    end do

  end subroutine backward
!!!#############################################################################


!!!#############################################################################
!!! update the weights based on how much error the node ...
!!! ... is responsible for
!!!#############################################################################
  subroutine update_weights_and_biases(learning_rate, gradients, clip, iteration)
    implicit none
    integer, optional, intent(inout) :: iteration
    real(real12), intent(in) :: learning_rate
    type(gradient_type), dimension(0:), intent(inout) :: gradients
    type(clip_type), optional, intent(in) :: clip
    
    integer :: j,k,l
    integer :: num_layers, num_neurons, num_inputs
    real(real12) :: weight_incr, rtmp1, rtmp2
    real(real12), allocatable, dimension(:) :: new_input


    !! apply gradient clipping
    if(present(clip))then
       if(clip%l_min_max) call gradient_clip(gradients,&
            clip_min=clip%min,clip_max=clip%max)
       if(clip%l_norm) call gradient_clip(gradients,&
            clip_norm=clip%norm)
    end if
    
    !! initialise constants
    num_layers=size(network,dim=1)

    !! loop through the layers in reverse
    do l=1,num_layers,1
       num_inputs  = size(network(l)%neuron(1)%weight, dim=1)
       num_neurons = size(network(l)%neuron, dim=1)
       do j=1,size(network(l)%neuron)
          !! update the weights and biases for layer l
          do k=1,num_inputs
             if(adaptive_parameters%method.eq.'none')then
                network(l)%neuron(j)%weight(k) = &
                     network(l)%neuron(j)%weight(k) - &
                     learning_rate * gradients(l)%weight(k,j)
             elseif(adaptive_parameters%method.eq.'adam')then
                call update_weight(learning_rate,&
                     network(l)%neuron(j)%weight(k),&
                     network(l)%neuron(j)%weight_incr(k), &
                     gradients(l)%weight(k,j), &
                     gradients(l)%m(k,j), &
                     gradients(l)%v(k,j), &
                     iteration, &
                     adaptive_parameters)
             else
                call update_weight(learning_rate,&
                     network(l)%neuron(j)%weight(k),&
                     network(l)%neuron(j)%weight_incr(k), &
                     gradients(l)%weight(k,j), &
                     rtmp1, &
                     rtmp2, &
                     iteration, &
                     adaptive_parameters)
             end if
          end do

       end do
    end do

    if(present(iteration)) iteration = iteration + 1


  end subroutine update_weights_and_biases
!!!#############################################################################


!!!#############################################################################
!!! gradient clipping
!!!#############################################################################
  subroutine gradient_clip(gradients,clip_min,clip_max,clip_norm)
    implicit none
    real(real12), optional, intent(in) :: clip_min, clip_max, clip_norm
    type(gradient_type), dimension(0:), intent(inout) :: gradients

    integer :: j, k, l, num_layers, num_neurons, num_inputs
    real(real12) :: norm

    !! clipping is not applied to deltas
    num_layers = ubound(gradients,dim=1)
    if(present(clip_norm))then
       gradients(l)%delta = gradients(l)%delta  * clip_norm/norm
       do l=1,num_layers
          norm = norm2(gradients(l)%weight(:,:))
          if(norm.gt.clip_norm)then
             gradients(l)%weight = &
                  gradients(l)%weight * clip_norm/norm
          end if
       end do
    elseif(present(clip_min).and.present(clip_max))then
       do l=1,num_layers
          num_inputs  = size(gradients(l)%weight,dim=1)
          num_neurons = size(gradients(l)%weight,dim=2)
          do j=1,num_neurons
             do k=1,num_inputs
                gradients(l)%weight(k,j) = &
                     max(clip_min,min(clip_max,gradients(l)%weight(k,j)))
             end do
          end do
       end do
    end if

  end subroutine gradient_clip
!!!#############################################################################


!!!#############################################################################
!!! activation function
!!! ... weight * input (basically, how much a 'parent' node influences a ...
!!! ... 'child' node
!!!#############################################################################
  function activate(weights, inputs) result(activation)
    implicit none
    integer :: i, num_weights
    real(real12), dimension(:), intent(in) :: weights
    real(real12), dimension(:), intent(in) :: inputs
    real(real12) :: activation
    
    num_weights = size(weights,dim=1)
    activation = weights(num_weights) !! starts with the bias
    !! then adds all of the weights multiplied by their respective inputs
    do i=1,num_weights-1
       activation = activation + weights(i) * inputs(i)
    end do
       
  end function activate
!!!#############################################################################


end module FullyConnectedLayer
