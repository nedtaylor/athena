!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module FullyConnectedLayer
  use constants, only: real12
  use misc_ml, only: adam_optimiser
  use custom_types, only: network_type, clip_type, activation_type, &
       learning_parameters_type
  use activation_gaussian, only: gaussian_setup
  use activation_linear, only: linear_setup
  use activation_piecewise, only: piecewise_setup
  use activation_relu, only: relu_setup
  use activation_leaky_relu, only: leaky_relu_setup
  use activation_sigmoid, only: sigmoid_setup
  use activation_tanh, only: tanh_setup
  use weight_initialiser, only: he_uniform, zeros
  implicit none


  type gradient_type
     real(real12), allocatable, dimension(:) :: delta !error !dE/d(activation)
     real(real12), allocatable, dimension(:,:) :: weight
     real(real12), allocatable, dimension(:,:) :: m
     real(real12), allocatable, dimension(:,:) :: v
   contains
     procedure :: add_t_t => gradient_add  !t = type, r = real, i = int
     generic :: operator(+) => add_t_t !, public
  end type gradient_type
  
  !type error_type
  !   real(real12), allocatable, dimension(:) :: neuron
  !end type error_type

  !interface sum_operator !operator(+)
  !   module procedure :: gradient_sum
  !end interface sum_operator !operator(+)

  type(network_type), allocatable, dimension(:) :: network
  type(learning_parameters_type) :: adaptive_parameters

  class(activation_type), allocatable :: transfer!activation


  private

  public :: network
  public :: gradient_type

  public :: initialise, forward, backward
  public :: update_weights_and_biases
  public :: write_file
  public :: normalise_delta_batch, reset_delta_batch
  public :: initialise_gradients


contains

!!!#############################################################################
!!! custom operation for summing gradient_type
!!!#############################################################################
  elemental function gradient_add(a, b) result(output)
    class(gradient_type), intent(in) :: a,b
    type(gradient_type) :: output
    
    if(allocated(a%weight).and.allocated(b%weight))then
       !allocate(output%weight,mold=a%weight)
       output%weight = a%weight + b%weight
       !!output%delta = output%delta + input%delta
       if(allocated(a%m)) output%m = a%m !+ input%m
       if(allocated(a%v)) output%v = a%v !+ input%v
    end if

  end function gradient_add
!!!#############################################################################


!!!!!#############################################################################
!!!! custom operation for summing gradient_type
!!!!#############################################################################
!  subroutine gradient_sum(output, input)
!    type(gradient_type), dimension(0:), intent(in) :: input
!    type(gradient_type), dimension(0:), intent(inout) :: output
!    integer :: i
!    
!    do i=1,ubound(output, dim=1) !! ignore 0 as that is a delta, which is not summed
!       output(i) = output(i) + input(i)
!    end do
!
!  end subroutine gradient_sum
!!!!#############################################################################


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

    integer :: l,i



    !! if file, read in weights and biases
    !! ... if no file is given, weights and biases to a default
    if(present(file))then
       !!-----------------------------------------------------------------------
       !! read convolution layer data from file
       !!-----------------------------------------------------------------------
       call read_file(file)
       return
    elseif(present(num_layers).and.present(num_inputs).and.&
         present(num_hidden))then
       !!-----------------------------------------------------------------------
       !! initialise random seed
       !!-----------------------------------------------------------------------
       call random_seed(size=nseed)
       allocate(seed_arr(nseed))
       if(present(seed))then
          seed_arr = seed
       else
          call system_clock(count=itmp1)
          seed_arr = itmp1 + 37 * (/ (l-1,l=1,nseed) /)
       end if
       call random_seed(put=seed_arr)

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
             allocate(network(l)%neuron(i)%weight(length))
             call random_number(network(l)%neuron(i)%weight)
             allocate(network(l)%neuron(i)%weight_incr(length))
             network(l)%neuron(i)%weight_incr = 0._real12
             network(l)%neuron(i)%output = 0._real12


!!! CALL A GENERAL FUNCTION THAT PASES TO THE INTERNAL SUBROUTINE BASED ON THE CASE
             call he_uniform(network(l)%neuron(i)%weight, &
                  size(network(l)%neuron(i)%weight,dim=1))

          end do

       end do
    else
       write(0,*) "ERROR: Not enough optional arguments provided to initialse FC"
       write(0,*) "Either provide (file) or (num_layers, num_inputs, and num_hidden)"
       write(0,*) "... seed is also optional for the latter set)"
       write(0,*) "Exiting..."
       stop
    end if

    
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
    select case(trim(t_activation_function))
    case("gaussian")
       transfer = gaussian_setup(scale = scale)
    case ("linear")
       transfer = linear_setup(scale = scale)
    case ("piecewise")
       transfer = piecewise_setup(scale = scale)
    case ("relu")
       transfer = relu_setup(scale = scale)
    case ("leaky_relu")
       transfer = leaky_relu_setup(scale = scale)
    case ("sigmoid")
       transfer = sigmoid_setup(scale = scale)
    case ("tanh")
       transfer = tanh_setup(scale = scale)
    case default
       transfer = relu_setup(scale = scale)
    end select
 

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
!!! 
!!!#############################################################################
  subroutine initialise_gradients(gradients, num_features, adam_learning)
    implicit none
    logical, optional, intent(in) :: adam_learning
    integer, intent(in) :: num_features
    type(gradient_type), allocatable, dimension(:), intent(out) :: gradients
    integer :: l
    integer :: num_neurons, num_inputs

    allocate(gradients(0:size(network,dim=1)))
    allocate(gradients(0)%delta(num_features))
    gradients(0)%delta = 0._real12

    do l=1,size(network,dim=1)
       num_neurons = size(network(l)%neuron,dim=1)
       num_inputs = size(network(l)%neuron(1)%weight,dim=1)
       allocate(gradients(l)%weight(num_inputs, num_neurons))
       allocate(gradients(l)%delta(num_neurons))
       gradients(l)%weight = 0._real12
       gradients(l)%delta  = 0._real12

       if(present(adam_learning))then
          if(adam_learning)then
             allocate(gradients(l)%m(num_inputs, num_neurons))
             allocate(gradients(l)%v(num_inputs, num_neurons))
             gradients(l)%m = 0._real12
             gradients(l)%v = 0._real12
          end if
       end if
    end do
    

  end subroutine initialise_gradients
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


    !! if file, read in weights and biases
    if(len(trim(file)).gt.0)then
       unit = 10
       found = .false.
       open(unit, file=trim(file))
       do while (.not.found)
          read(unit,'(A)',iostat=stat) buffer
          if(stat.ne.0)then
             write(0,*) "ERROR: file hit error (EoF?) before FULL_LAYER section"
             write(0,*) "Exiting..."
             stop
          end if
          if(trim(adjustl(buffer)).eq."FULL_LAYER") found = .true.
       end do

       !read(unit,*) input_size, output_size, num_layers
       read(unit,*)

       completed = 0
       !do while (completed.lt.2)
       !
       !   read(unit,'(A)',iostat=stat) buffer
       !   if(stat.ne.0)then
       !      write(0,*) "ERROR: file hit error (EoF?) before encountering END FULL_LAYER"
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
       !         if(i.gt.input_size)then
       !            write(0,*) "ERROR: i exceeded kernel_size in FULL_LAYER"
       !            write(0,*) "Exiting..."
       !            stop
       !         end if
       !         read(buffer,*) (weights(i,j),j=1,output_size)
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
       !         read(buffer,*) (biases(j),j=1,output_size)
       !      end do
       !   end if
       !end do
       close(unit)

       return
    end if

  end  subroutine read_file
!!!#############################################################################


!!!#############################################################################
!!! 
!!!#############################################################################
  subroutine write_file(file)
    implicit none
    character(*), intent(in) :: file

    integer :: num_layers,num_inputs,num_weights
    integer :: l,i
    integer :: unit=10
    character(128) :: fmt
    integer, allocatable, dimension(:) :: num_hidden
    real(real12), allocatable, dimension(:) :: biases

    open(unit, file=trim(file), access='append')

    num_layers = size(network,dim=1)
    num_inputs = size(network(1)%neuron(1)%weight,dim=1)

    allocate(num_hidden(num_layers))
    do l=1,num_layers
       num_hidden(l) = size(network(l)%neuron,dim=1)
    end do
    write(unit,'("FULLYCONNECTED")')
    !write(unit,'(3X,"NUM_LAYERS = ")') num_layers
    write(fmt,'("(3X,""LAYER_SIZES ="",",I0,"(1X,I0))")') num_layers
    write(unit,trim(fmt)) num_hidden(:)-1

    write(unit,'("BIASES")')
    do l=1,num_layers
       allocate(biases(num_hidden(l)))
       biases = 0._real12
       do i=1,num_hidden(l)
          num_weights = size(network(l)%neuron(i)%weight,dim=1)
          biases(i) = network(l)%neuron(i)%weight(num_weights)
       end do
       write(unit,*) biases(:)
       deallocate(biases)
    end do
    write(unit,'("END BIASES")')

    write(unit,'("WEIGHTS")')
    do l=1,num_layers
       do i=1,num_hidden(l)
          num_weights = size(network(l)%neuron(i)%weight,dim=1)
          write(unit,*) network(l)%neuron(i)%weight(:num_weights-1)
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
    real(real12), dimension(:), intent(out) :: output

    integer :: j, l
    integer :: num_layers, num_neurons
    real(real12) :: activation
    real(real12), allocatable, dimension(:) :: new_input
    
    allocate(new_input(size(input)))
    new_input = input
    num_layers = size(network,dim=1)
    do l=1,num_layers
       
       num_neurons=size(network(l)%neuron)
       do j=1,num_neurons
          activation = activate(network(l)%neuron(j)%weight,new_input)
          network(l)%neuron(j)%output = transfer%activate(activation)
       end do
       deallocate(new_input)
       if(l.lt.num_layers)then
          allocate(new_input(num_neurons))
          new_input = network(l)%neuron(:)%output
       end if
    end do
    output = network(num_layers)%neuron(:)%output

  end subroutine forward
!!!#############################################################################


!!!#############################################################################
!!! backward propagation
!!! method : gradient descent
!!!#############################################################################
!!! https://brilliant.org/wiki/backpropagation/
  subroutine backward(input, expected, input_gradients, clip)
    implicit none
    real(real12), dimension(:), intent(in) :: input
    real(real12), dimension(:), intent(in) :: expected !is this just output_gradients?
    type(gradient_type), dimension(0:), intent(inout) :: input_gradients
    type(clip_type), optional, intent(in) :: clip
    
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
             new_input = network(l-1)%neuron(:)%output
          end if
          do j=1,num_neurons
             !! activation already calculated and equals the output
             input_gradients(l)%delta(j) = input_gradients(l)%delta(j) * &
                  transfer%differentiate(network(l)%neuron(j)%output)
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
!!! update the weights based on how much error the node ...
!!! ... is responsible for
!!!#############################################################################
  subroutine update_weights_and_biases(learning_rate, input, gradients, &
       l1_lambda, l2_lambda, iteration)
    implicit none
    integer, optional, intent(inout) :: iteration
    real(real12), optional, intent(in) :: l1_lambda, l2_lambda
    real(real12), intent(in) :: learning_rate
    real(real12), dimension(:), intent(in) :: input
    type(gradient_type), dimension(0:), intent(inout) :: gradients
    
    integer :: j,k,l
    integer :: num_layers, num_weights, num_neurons, num_inputs
    real(real12) :: t_learning_rate
    real(real12) :: lr_l1_lambda, lr_l2_lambda, weight_incr
    real(real12), allocatable, dimension(:) :: new_input

    
    num_layers=size(network,dim=1)
    lr_l1_lambda = learning_rate * l1_lambda
    lr_l2_lambda = learning_rate * l2_lambda


    !! loop through the layers in reverse
    do l=1,num_layers,1
       !! inputs are equal to the ouputs of the 'parent' nodes
       if (l.ne.1)then
          allocate(new_input(num_neurons))
          new_input = network(l-1)%neuron(:)%output
       else
          allocate(new_input(size(input)))
          new_input = input
       end if
       num_inputs = size(new_input, dim=1)
       num_weights = size(network(l)%neuron(1)%weight, dim=1)
       num_neurons = size(network(l)%neuron, dim=1)
       do j=1,size(network(l)%neuron)
          !! for each path, update the weight based on the delta ...
          !! ... and the learning rate

          do k=1,num_inputs
             
             t_learning_rate = learning_rate
             weight_incr = network(l)%neuron(j)%weight_incr(k) 

             !! momentum-based learning
             !! adam optimiser
             if(adaptive_parameters%method.eq.'momentum')then
                weight_incr = learning_rate * gradients(l)%weight(k,j) + &
                     adaptive_parameters%momentum * weight_incr
             elseif(adaptive_parameters%method.eq.'adam')then
                call adam_optimiser(t_learning_rate, gradients(l)%weight(k,j), &
                     gradients(l)%m(k,j), gradients(l)%v(k,j), iteration, &
                     adaptive_parameters%beta1, adaptive_parameters%beta2, &
                     adaptive_parameters%epsilon)
                weight_incr = t_learning_rate !* new_input(k) !! unsure about new_input here
                !write(0,*) "HERE", weight_incr
             else
                weight_incr = learning_rate * gradients(l)%weight(k,j)
             end if

             !! L1 regularisation
             if(present(l1_lambda))then
                !network(l)%neuron(j)%weight(k) = network(l)%neuron(j)%weight(k) - &
                !     learning_rate * l1_lambda * sign(1._real12,network(l)%neuron(j)%weight(k))
                weight_incr = weight_incr + &
                     lr_l1_lambda * sign(1._real12,network(l)%neuron(j)%weight(k))
             end if

             !! L2 regularisation
             if(present(l2_lambda))then
                !network(l)%neuron(j)%weight(k) = network(l)%neuron(j)%weight(k) * &
                !     (1._real12 - learning_rate * l2_lambda)
                weight_incr = weight_incr + &
                     lr_l2_lambda * network(l)%neuron(j)%weight(k)
             end if


             network(l)%neuron(j)%weight_incr(k) = weight_incr
             network(l)%neuron(j)%weight(k) = network(l)%neuron(j)%weight(k) - &
                  weight_incr

          end do

          !! update biases
          network(l)%neuron(j)%weight(num_weights) = &
               network(l)%neuron(j)%weight(num_weights) - &
               learning_rate * gradients(l)%weight(num_weights,j)
       end do
       deallocate(new_input)
    end do

    if(present(iteration)) iteration = iteration + 1


  end subroutine update_weights_and_biases
!!!#############################################################################


!!!#############################################################################
!!! 
!!!#############################################################################
  subroutine normalise_delta_batch(batch_size)
    implicit none
    integer :: l
    integer, intent(in) :: batch_size

    do l=1,size(network,dim=1)
       network(l)%neuron(:)%delta_batch = network(l)%neuron(:)%delta_batch/batch_size
    end do

  end subroutine normalise_delta_batch
!!!#############################################################################


!!!#############################################################################
!!! 
!!!#############################################################################
  subroutine reset_delta_batch()
    implicit none
    integer :: l

    do l=1,size(network,dim=1)
       network(l)%neuron(:)%delta_batch = 0._real12
    end do

  end subroutine reset_delta_batch
!!!#############################################################################


!!!#############################################################################
!!! gradient clipping
!!!#############################################################################
  subroutine gradient_clip(gradients,clip_min,clip_max,clip_norm)
    implicit none
    real(real12), optional, intent(in) :: clip_min, clip_max, clip_norm
    type(gradient_type), dimension(0:), intent(inout) :: gradients

    integer :: j, k, l, num_layers, num_neurons
    real(real12) :: norm

    num_layers = ubound(gradients,dim=1)
    if(present(clip_norm))then
       do l=1,num_layers
          norm = norm2(gradients(l)%weight(:,:))
          if(norm.gt.clip_norm)then
             gradients(l)%weight = &
                  gradients(l)%weight * clip_norm/norm
             !gradients(l)%delta = gradients(l)%delta  * clip_norm/norm
          end if
       end do
    elseif(present(clip_min).and.present(clip_max))then
       do l=1,num_layers
          num_neurons = size(gradients(l)%weight)
          do j=1,num_neurons
             do k=1,size(gradients(l)%weight,dim=1)
                gradients(l)%weight(k,j) = &
                     max(clip_min,min(clip_max,gradients(l)%weight(k,j)))
             end do
             !gradients(l)%delta(j) = &
             !     max(clip_min,min(clip_max,gradients(l)%delta(j)))
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
