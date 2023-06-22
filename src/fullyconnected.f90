!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module FullyConnectedLayer
  use constants, only: real12
  use custom_types, only: network_type, clip_type
  implicit none

  type(network_type), allocatable, dimension(:) :: network

  private

  public :: network
  public :: initialise, forward, backward
  public :: update_weights_and_biases
  public :: write_file
  public :: normalise_delta_batch, reset_delta_batch


contains

!!!#############################################################################
!!!
!!!#############################################################################
  subroutine initialise(seed, num_layers, num_inputs, num_hidden, file)
    implicit none
    integer, intent(in), optional :: seed
    integer, intent(in), optional :: num_layers, num_inputs
    integer, dimension(:), intent(in), optional :: num_hidden
    character(*), optional, intent(in) :: file

    integer :: itmp1,nseed
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
          seed_arr = itmp1 + 37* (/ (l-1,l=1,nseed) /)
       end if
       call random_seed(put=seed_arr)

       !!-----------------------------------------------------------------------
       !! randomly initialise convolution layers
       !!-----------------------------------------------------------------------
       allocate(network(num_layers))
       do l=1,num_layers
          allocate(network(l)%neuron(num_hidden(l)))
          if(l.eq.1)then
             do i=1,num_hidden(l)
                allocate(network(l)%neuron(i)%weight(num_inputs+1))
                call random_number(network(l)%neuron(i)%weight)
                allocate(network(l)%neuron(i)%weight_incr(num_inputs+1))
                network(l)%neuron(i)%weight_incr = 0._real12
             end do
          else
             do i=1,num_hidden(l)
                allocate(network(l)%neuron(i)%weight(num_hidden(l-1)+1))
                call random_number(network(l)%neuron(i)%weight)
                allocate(network(l)%neuron(i)%weight_incr(num_inputs+1))
                network(l)%neuron(i)%weight_incr = 0._real12
                network(l)%neuron(i)%delta = 0._real12
                network(l)%neuron(i)%delta_batch = 0._real12
                network(l)%neuron(i)%output = 0._real12
             end do
          end if
       end do
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
    real(real12), allocatable, dimension(:) :: new_inputs
    
    !output = forward_propagate(network, input)

    allocate(new_inputs(size(input)))
    new_inputs = input
    num_layers=size(network)
    do l=1,num_layers
       
       num_neurons=size(network(l)%neuron)
       do j=1,num_neurons
          activation = activate(network(l)%neuron(j)%weight,new_inputs)
          network(l)%neuron(j)%output = relu(activation)
       end do
       deallocate(new_inputs)
       if(l.lt.num_layers)then
          allocate(new_inputs(num_neurons))
          new_inputs = network(l)%neuron(:)%output
       end if
    end do
    output = network(num_layers)%neuron(:)%output

  end subroutine forward
!!!#############################################################################


!!!#############################################################################
!!! 
!!!#############################################################################
  subroutine backward(input, expected, input_gradients, clip)
    implicit none
    real(real12), dimension(:), intent(in) :: input
    real(real12), dimension(:), intent(in) :: expected !is this just output_gradients?
    real(real12), dimension(:), intent(out) :: input_gradients
    type(clip_type), optional, intent(in) :: clip
    
    integer :: j, k, l
    integer :: num_layers, num_neurons
    real(real12), allocatable, dimension(:) :: errors

    !! Initialise input_gradients to zero
    input_gradients = 0._real12
    num_layers=size(network,dim=1)

    !write(0,*) "delta start"
    !write(0,*) network(num_layers)%neuron(:)%delta


    !! loop through the layers in reverse
    do l=num_layers,1,-1
       num_neurons = size(network(l)%neuron)
       allocate(errors(num_neurons))
       errors = 0._real12
       !! for first layer, the error is the difference between ...
       !! ... predicted and expected
       if (l.eq.num_layers)then
          !errors(:) = network(l)%neuron(:)%output - expected
          errors(:) = expected
          !write(0,*) "output", network(l)%neuron(:)%output
          !write(0,*) "expected", expected
          !write(0,*) "errors", errors
       else
          do j=1,num_neurons
             !! the errors are summed from the delta of the ...
             !! ... 'child' node * 'child' weight
             do k=1,size(network(l+1)%neuron)
                errors(j) = errors(j) + (network(l+1)%neuron(k)%weight(j) * network(l+1)%neuron(k)%delta)
             end do
          end do
       end if
       !! the delta values are the error multipled by the derivative ...
       !! ... of the transfer function
       do j=1,num_neurons
          network(l)%neuron(j)%delta = errors(j) * relu_derivative(network(l)%neuron(j)%output)
       end do
       deallocate(errors)
    end do

    !! apply gradient clipping
    if(present(clip))then
       if(clip%l_min_max) call gradient_clip(&
            clip_min=clip%min,clip_max=clip%max)
       if(clip%l_norm) call gradient_clip(&
            clip_norm=clip%norm)
    end if

    !! mini batch gradients
    do l=1,num_layers
       num_neurons = size(network(l)%neuron)
       do j=1,num_neurons
          network(l)%neuron(j)%delta_batch = network(l)%neuron(j)%delta_batch + network(l)%neuron(j)%delta
       end do
    end do

    !! store gradients in input_gradients
    !! ... not currently used
    input_gradients = network(1)%neuron(:)%delta

  end subroutine backward
!!!#############################################################################


!!!#############################################################################
!!! update the weights based on how much error the node ...
!!! ... is responsible for
!!!#############################################################################
  subroutine update_weights_and_biases(learning_rate, input, gradients, &
       l1_lambda, l2_lambda, momentum, l_batch)
    implicit none
    real(real12), optional, intent(in) :: l1_lambda, l2_lambda, momentum
    real(real12), intent(in) :: learning_rate
    real(real12), dimension(:), intent(in) :: input
    real(real12), dimension(:), optional, intent(in) :: gradients
    logical, optional, intent(in) :: l_batch

    integer :: j,k,l
    integer :: num_layers,num_weights
    real(real12) :: gradient
    logical :: use_batch
    real(real12), allocatable, dimension(:) :: new_inputs

    
!!! GRADIENTS NOT YET USED AS DELTA TAKES ITS PLACE
    use_batch=.false.
    if(present(l_batch))then
       use_batch=l_batch
    end if
    num_layers=size(network)

    !write(0,*) network(num_layers)%neuron(:)%delta_batch
    !write(0,*) network(num_layers-1)%neuron(:)%output
    !write(0,*) input!network(1)%neuron(:)%output
    !write(0,*) (network(1)%neuron(j)%weight(:),j=1,size(network(1)%neuron,1))

    !! loop through the layers in reverse
    do l=1,num_layers,1
       !! inputs are equal to the ouputs of the 'parent' nodes
       if (l.ne.1)then
          allocate(new_inputs(size(network(l-1)%neuron)))
          do j=1,size(network(l-1)%neuron)
             new_inputs(j) = network(l-1)%neuron(j)%output
          end do
       else
          allocate(new_inputs(size(input)))
          new_inputs = input
       end if
       do j=1,size(network(l)%neuron)
          !! for each path, update the weight based on the delta ...
          !! ... and the learning rate

          if(use_batch)then
             gradient = network(l)%neuron(j)%delta_batch
          else
             gradient = network(l)%neuron(j)%delta
          end if
          
          do k=1,size(new_inputs)

             !! momentum-based learning
             if(present(momentum))then
                
                network(l)%neuron(j)%weight_incr(k) = &
                     learning_rate * gradient * new_inputs(k) + &
                     momentum * network(l)%neuron(j)%weight_incr(k)

                !network(l)%neuron(j)%weight(k) = network(l)%neuron(j)%weight(k) - &
                !     network(l)%neuron(j)%weight_incr(k)
             else
                !network(l)%neuron(j)%weight(k) = network(l)%neuron(j)%weight(k) - &
                !     learning_rate * network(l)%neuron(j)%delta_batch * new_inputs(k)
                network(l)%neuron(j)%weight_incr(k) = &
                     learning_rate * gradient * new_inputs(k)
             end if

             !! L1 regularisation
             if(present(l1_lambda))then
                !network(l)%neuron(j)%weight(k) = network(l)%neuron(j)%weight(k) - &
                !     learning_rate * l1_lambda * sign(1._real12,network(l)%neuron(j)%weight(k))
                network(l)%neuron(j)%weight_incr(k) = &
                     network(l)%neuron(j)%weight_incr(k) + &
                     learning_rate * l1_lambda * sign(1._real12,network(l)%neuron(j)%weight(k))
             end if

             !! L2 regularisation
             if(present(l2_lambda))then
                !network(l)%neuron(j)%weight(k) = network(l)%neuron(j)%weight(k) * &
                !     (1._real12 - learning_rate * l2_lambda)
                network(l)%neuron(j)%weight_incr(k) = &
                     network(l)%neuron(j)%weight_incr(k) + &
                     learning_rate * l2_lambda * network(l)%neuron(j)%weight(k)
             end if


             network(l)%neuron(j)%weight(k) = network(l)%neuron(j)%weight(k) - &
                  network(l)%neuron(j)%weight_incr(k)




          end do
          num_weights = size(network(l)%neuron(j)%weight)
          network(l)%neuron(j)%weight(num_weights) = &
               network(l)%neuron(j)%weight(num_weights) - &
               learning_rate * gradient
       end do
       deallocate(new_inputs)
    end do


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
  subroutine gradient_clip(clip_min,clip_max,clip_norm)
    implicit none
    real(real12), optional, intent(in) :: clip_min, clip_max, clip_norm

    integer :: j, l, num_layers, num_neurons
    real(real12) :: norm

    num_layers = size(network,dim=1)
    if(present(clip_norm))then
       do l=1,num_layers
          norm = norm2(network(l)%neuron(:)%delta)
          if(norm.gt.clip_norm)then
             network(l)%neuron(:)%delta = &
                  network(l)%neuron(:)%delta * clip_norm/norm
          end if
       end do
    elseif(present(clip_min).and.present(clip_max))then
       do l=1,num_layers
          num_neurons = size(network(l)%neuron)   
          do j=1,num_neurons
             network(l)%neuron(j)%delta = &
                  max(clip_min,min(clip_max,network(l)%neuron(j)%delta))
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
    integer :: i, n_weights
    real(real12), dimension(:), intent(in) :: weights
    real(real12), dimension(:), intent(in) :: inputs
    real(real12) :: activation
    
    n_weights = size(weights,dim=1)
    activation = weights(n_weights) !! starts with the bias
    !! then adds all of the weights multiplied by their respective inputs
    do i=1,n_weights-1
       activation = activation + weights(i) * inputs(i)
    end do
       
  end function activate
!!!#############################################################################


!!!#############################################################################
!!! Linear transfer function
!!! f = gradient * x
!!!#############################################################################
  function linear(val, gradient) result(output)
    implicit none
    real(real12), intent(in) :: val
    real(real12), optional, intent(in) :: gradient
    real(real12) :: output

    if(present(gradient))then
       output = gradient * val
    else
       output = 0.05_real12 * val
    end if
  end function linear
!!!#############################################################################


!!!#############################################################################
!!! RELU transfer function
!!! f = max(0, x)
!!!#############################################################################
  function relu(val) result(output)
    implicit none
    real(real12), intent(in) :: val
    real(real12) :: output

    output = max(0._real12, val)
  end function relu
!!!#############################################################################


!!!#############################################################################
!!! leaky RELU transfer function
!!! f = max(0, x)
!!!#############################################################################
  function leaky_relu(val) result(output)
    implicit none
    real(real12), intent(in) :: val
    real(real12) :: output

    output = max(0.01_real12*val, val)
  end function leaky_relu
!!!#############################################################################


!!!#############################################################################
!!! sigmoid transfer function
!!! f = 1/(1+exp(-x))
!!!#############################################################################
  function sigmoid(val) result(output)
    implicit none
    real(real12), intent(in) :: val
    real(real12) :: output

    output = 1._real12 /(1._real12 + exp(-val))
  end function sigmoid
!!!#############################################################################


!!!#############################################################################
!!! tanh transfer function
!!! f = (exp(x) - exp(-x))/(exp(x) + exp(-x))
!!!#############################################################################
  function tanh(val) result(output)
    implicit none
    real(real12), intent(in) :: val
    real(real12) :: output

    output = (exp(val) - exp(-val))/(exp(val) + exp(-val))
  end function tanh
!!!#############################################################################
    

!!!#############################################################################
!!! derivative of RELU transfer function
!!! e.g. df/dx (gradient * x) = gradient
!!! we are performing the derivative to identify what weight ...
!!! ... results in the minimum error
!!!#############################################################################
  function linear_derivative(val, gradient) result(derivative)
    implicit none
    real(real12), intent(in) :: val
    real(real12), optional, intent(in) :: gradient
    real(real12) :: derivative

    if(present(gradient))then
       derivative = gradient * val
    else
       derivative = 0.05_real12 * val
    end if
  end function linear_derivative
!!!#############################################################################


!!!#############################################################################
!!! derivative of RELU transfer function
!!! e.g. df/dx (1*x) = 1
!!! we are performing the derivative to identify what weight ...
!!! ... results in the minimum error
!!!#############################################################################
  function relu_derivative(val) result(derivative)
    implicit none
    real(real12), intent(in) :: val
    real(real12) :: derivative

    if(val.ge.0._real12)then
       derivative = 1._real12
    else
       derivative = 0._real12
    end if
  end function relu_derivative
!!!#############################################################################


!!!#############################################################################
!!! derivative of RELU transfer function
!!! e.g. df/dx (0.05*x) = 0.05
!!! we are performing the derivative to identify what weight ...
!!! ... results in the minimum error
!!!#############################################################################
  function leaky_relu_derivative(val) result(derivative)
    implicit none
    real(real12), intent(in) :: val
    real(real12) :: derivative

    if(val.ge.0._real12)then
       derivative = 1._real12
    else
       derivative = 0.01_real12
    end if
  end function leaky_relu_derivative
!!!#############################################################################


!!!#############################################################################
!!! derivative of sigmoid function
!!!#############################################################################
  function sigmoid_derivative(val) result(output)
    implicit none
    real(real12), intent(in) :: val
    real(real12) :: output

    output = sigmoid(val) * (1._real12 - sigmoid(val))
  end function sigmoid_derivative
!!!#############################################################################


!!!#############################################################################
!!! derivative of tanh function
!!!#############################################################################
  function tanh_derivative(val) result(output)
    implicit none
    real(real12), intent(in) :: val
    real(real12) :: output

    output = 1._real12 - tanh(val) ** 2._real12
  end function tanh_derivative
!!!#############################################################################

end module FullyConnectedLayer
