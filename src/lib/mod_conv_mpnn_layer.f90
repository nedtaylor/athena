!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
module conv_mpnn_layer
  use constants, only: real12
  use misc, only: outer_product
  use custom_types, only: graph_type, activation_type
  use activation_softmax, only: softmax_setup
  use mpnn_layer, only: &
       mpnn_layer_type, mpnn_method_type, &
       state_method_type, message_method_type, readout_method_type, &
       feature_type
  implicit none
  

  private
  public :: conv_mpnn_layer_type



  type, extends(message_method_type) :: convolutional_message_method_type
   contains
     procedure :: update => convolutional_message_update
     procedure :: get_differential => convolutional_get_message_differential
     procedure :: calculate_partials => convolutional_calculate_message_partials
  end type convolutional_message_method_type
  interface convolutional_message_method_type
    module function message_method_setup( &
         num_vertex_features, num_edge_features, batch_size ) result(message_method)
      integer, intent(in) :: num_vertex_features, num_edge_features, batch_size
      type(convolutional_message_method_type) :: message_method
    end function message_method_setup
  end interface convolutional_message_method_type


  type, extends(state_method_type) :: convolutional_state_method_type
     !! weight has dimensions (feature_out, feature_in, vertex_degree, batch_size)
     real(real12), dimension(:,:,:), allocatable :: weight
     real(real12), dimension(:,:,:,:), allocatable :: dw
     type(feature_type), dimension(:), allocatable :: z
     class(activation_type), allocatable :: transfer
   contains
     procedure :: update => convolutional_state_update
     procedure :: get_differential => convolutional_get_state_differential
     procedure :: calculate_partials => convolutional_calculate_state_partials
  end type convolutional_state_method_type
  interface convolutional_state_method_type
    module function state_method_setup( &
         num_vertex_features, num_edge_features, max_vertex_degree, batch_size ) result(state_method)
      integer, intent(in) :: num_vertex_features, num_edge_features, max_vertex_degree, batch_size
      type(convolutional_state_method_type) :: state_method
    end function state_method_setup
  end interface convolutional_state_method_type


  type, extends(readout_method_type) :: convolutional_readout_method_type
    integer :: num_time_steps
    real(real12), dimension(:,:,:), allocatable :: weight
    real(real12), dimension(:,:,:,:), allocatable :: dw
    type(feature_type), dimension(:,:), allocatable :: z
    class(activation_type), allocatable :: transfer
   contains
     procedure :: get_output => convolutional_get_readout_output
     procedure :: get_differential => convolutional_get_readout_differential
     procedure :: calculate_partials => convolutional_calculate_readout_partials
  end type convolutional_readout_method_type
  interface convolutional_readout_method_type
    module function readout_method_setup( &
         num_time_steps, num_inputs, num_outputs, batch_size ) result(readout_method)
      integer, intent(in) :: num_time_steps, num_inputs, num_outputs, batch_size
      type(convolutional_readout_method_type) :: readout_method
    end function readout_method_setup
  end interface convolutional_readout_method_type




  type, extends(mpnn_layer_type) :: conv_mpnn_layer_type
  end type conv_mpnn_layer_type

  type, extends(mpnn_method_type) :: conv_mpnn_method_type
   contains
    procedure, pass(this) :: init => init_conv_mpnn_method
  end type conv_mpnn_method_type


  interface conv_mpnn_layer_type
    module function layer_setup( &
           num_time_steps, &
           num_vertex_features, num_edge_features, &
           num_outputs, batch_size ) result(layer)
      integer, intent(in) :: num_time_steps, num_vertex_features, &
           num_edge_features, num_outputs, batch_size
      type(conv_mpnn_layer_type) :: layer
    end function layer_setup
  end interface conv_mpnn_layer_type




contains

  module function message_method_setup( &
         num_vertex_features, num_edge_features, batch_size ) result(message_method)
    implicit none
    integer, intent(in) :: num_vertex_features, num_edge_features, batch_size
    type(convolutional_message_method_type) :: message_method

    message_method%num_features = num_vertex_features + num_edge_features
    message_method%batch_size = batch_size
    
    allocate(message_method%feature(batch_size))
    allocate(message_method%di(batch_size))

  end function message_method_setup

  module function state_method_setup( &
         num_vertex_features, num_edge_features, max_vertex_degree, batch_size ) result(state_method)
    implicit none
    integer, intent(in) :: num_vertex_features, num_edge_features, batch_size, max_vertex_degree
    type(convolutional_state_method_type) :: state_method

    state_method%num_features = num_vertex_features
    state_method%batch_size = batch_size

    !!! MAXIMUM VERTEX DEGREE
    allocate(state_method%weight(num_vertex_features, num_vertex_features + num_edge_features, max_vertex_degree))
    allocate(state_method%dw(num_vertex_features, num_vertex_features + num_edge_features, max_vertex_degree, batch_size))
    allocate(state_method%z(batch_size))
  
  end function state_method_setup

  module function readout_method_setup( &
         num_time_steps, num_inputs, num_outputs,batch_size ) result(readout_method)
    implicit none
    integer, intent(in) :: num_time_steps, num_inputs, num_outputs, batch_size
    type(convolutional_readout_method_type) :: readout_method

    readout_method%num_time_steps = num_time_steps
    readout_method%num_outputs = num_outputs
    readout_method%batch_size = batch_size

    allocate(readout_method%weight(num_outputs, num_inputs, num_time_steps))
    allocate(readout_method%dw(num_outputs, num_time_steps, num_outputs, batch_size))
    allocate(readout_method%z(num_time_steps, batch_size))

  end function readout_method_setup



  subroutine init_conv_mpnn_method(this, input_shape, output_shape, batch_size)
    implicit none
    class(conv_mpnn_method_type), intent(inout) :: this
    integer, dimension(3), intent(in) :: input_shape
    integer, dimension(1), intent(in) :: output_shape
    integer, intent(in) :: batch_size

    integer :: t

    this%num_features = input_shape(:2)
    this%num_time_steps = input_shape(3)
    this%num_outputs = output_shape(1)
    allocate(this%message(this%num_time_steps))
    allocate(this%state(this%num_time_steps))
    allocate(this%message(this%num_time_steps), &
         source = convolutional_message_method_type( &
            this%num_features(1), this%num_features(2), batch_size &
         ) &
    )
    allocate(this%state(this%num_time_steps), &
         source = convolutional_state_method_type( &
            this%num_features(1), this%num_features(2), 4, batch_size &
         ) &
    )
    allocate(this%readout, &
         source = convolutional_readout_method_type( &
              this%num_time_steps, this%num_outputs, this%num_outputs, batch_size &
         ) &
    )

  end subroutine init_conv_mpnn_method



  module function layer_setup( &
         num_time_steps, &
         num_vertex_features, num_edge_features, &
         num_outputs, batch_size ) result(layer)
    implicit none
    integer, intent(in) :: num_time_steps, num_vertex_features, &
         num_edge_features, num_outputs, batch_size
    type(conv_mpnn_layer_type) :: layer

    layer%batch_size = batch_size

    call layer%method%init(&
         [num_vertex_features, num_edge_features, num_time_steps], [num_outputs], batch_size)

    allocate(layer%output(num_outputs, layer%batch_size))

  end function layer_setup


  pure subroutine convolutional_message_update(this, input, graph)
    implicit none
    class(convolutional_message_method_type), intent(inout) :: this
    type(feature_type), dimension(this%batch_size), intent(in) :: input
    type(graph_type), dimension(this%batch_size), intent(in) :: graph

    integer :: s, v, w

    do s = 1, this%batch_size
       do v = 1, graph(s)%num_vertices
         this%feature(s)%val(:,v) = 0._real12
         do w = 1, graph(s)%num_vertices
             if(graph(s)%adjacency(v,w) .eq. 1) then
               this%feature(s)%val(:,v) = &
                     this%feature(s)%val(:,v) + &
                     [ input(s)%val(:,w), graph(s)%edge(v,w)%feature(:) ]
             end if
         end do
       end do
    end do

  end subroutine convolutional_message_update

  pure function convolutional_get_message_differential(this, input, graph) &
       result(output)
    implicit none
    class(convolutional_message_method_type), intent(in) :: this
    type(feature_type), dimension(this%batch_size), intent(in) :: input
    type(graph_type), dimension(this%batch_size), intent(in) :: graph
    
    type(feature_type), dimension(this%batch_size) :: output

    integer :: s, v

    do s = 1, this%batch_size
       do v = 1, graph(s)%num_vertices
          output(s)%val(:,v) = 1._real12
       end do
    end do

  end function convolutional_get_message_differential

  pure subroutine convolutional_calculate_message_partials(this, input, gradient, graph)
    implicit none
    class(convolutional_message_method_type), intent(inout) :: this
    !! hidden features has dimensions (feature, vertex, batch_size)
    type(feature_type), dimension(this%batch_size), intent(in) :: input
    type(feature_type), dimension(this%batch_size), intent(in) :: gradient
    type(graph_type), dimension(this%batch_size), intent(in) :: graph

    integer :: s

    !! the delta values are the error multipled by the derivative ...
    !! ... of the transfer function
    !! delta(l) = g'(a) * dE/dI(l)
    !! delta(l) = differential of activation * error from next layer

    do concurrent(s=1:this%batch_size)
       !! no message passing transfer function
       this%di(s)%val(:,:) = gradient(s)%val(:,:graph(s)%num_vertex_features)
    end do

  end subroutine convolutional_calculate_message_partials


  pure subroutine convolutional_state_update(this, input, graph)
    implicit none
    class(convolutional_state_method_type), intent(inout) :: this
    type(feature_type), dimension(this%batch_size), intent(in) :: input
    type(graph_type), dimension(this%batch_size), intent(in) :: graph

    integer :: s, v

    do s = 1, this%batch_size
       do v = 1, graph(s)%num_vertices
          this%feature(s)%val(:,v) = this%transfer%activate( &
                matmul( &
                     this%weight(:,:,graph(s)%vertex(v)%degree), &
                     input(s)%val(:,v) &
                ) )
       end do
    end do

  end subroutine convolutional_state_update

  pure function convolutional_get_state_differential(this, input, graph) result(output)
    implicit none
    class(convolutional_state_method_type), intent(in) :: this
    type(feature_type), dimension(this%batch_size), intent(in) :: input
    type(graph_type), dimension(this%batch_size), intent(in) :: graph

    type(feature_type), dimension(this%batch_size):: output

    integer :: v, s

    do s = 1, this%batch_size
       do v = 1, graph(s)%num_vertices
          output(s)%val(:,v) = this%transfer%differentiate( &
               matmul( &
                    this%weight(:,:,graph(s)%vertex(v)%degree), &
                    input(s)%val(:,v) &
               ) )
       end do
    end do

  end function convolutional_get_state_differential

  pure subroutine convolutional_calculate_state_partials(this, input, gradient, graph)
    implicit none
    class(convolutional_state_method_type), intent(inout) :: this
    type(feature_type), dimension(this%batch_size), intent(in) :: input
    type(feature_type), dimension(this%batch_size), intent(in) :: gradient
    type(graph_type), dimension(this%batch_size), intent(in) :: graph

    integer :: s, v, degree
    real(real12), dimension(:,:), allocatable :: delta

    
    !! the delta values are the error multipled by the derivative ...
    !! ... of the transfer function
    !! delta(l) = g'(a) * dE/dI(l)
    !! delta(l) = differential of activation * error from next layer


    do concurrent(s=1:this%batch_size)
       !! no message passing transfer function
       delta(:,:) = gradient(s)%val(:,:) * &
            this%transfer%differentiate(this%z(s)%val(:,:))
       
       !! partial derivatives of error wrt weights
       !! dE/dW = o/p(l-1) * delta
       do v = 1, graph(s)%num_vertices
          degree = graph(s)%vertex(v)%degree
          !! i.e. outer product of the input and delta
          !! sum weights and biases errors to use in batch gradient descent
          this%dw(:,:,degree,s) = this%dw(:,:,degree,s) + outer_product(input(s)%val(:,v), delta(:,v))
          !! the errors are summed from the delta of the ...
          !! ... 'child' node * 'child' weight
          !! dE/dI(l-1) = sum(weight(l) * delta(l))
          !! this prepares dE/dI for when it is passed into the previous layer
          this%di(s)%val(:,v) = matmul(this%weight(:,:,degree), delta(:,v))
       end do
    end do

  end subroutine convolutional_calculate_state_partials


  pure function convolutional_get_readout_output(this, input) result(output)
    implicit none
    class(convolutional_readout_method_type), intent(in) :: this
    class(state_method_type), dimension(:), intent(in) :: input

    real(real12), dimension(:,:), allocatable :: output

    integer :: s, v, t

    do s = 1, this%batch_size
       output(:,s) = 0._real12
       do v = 1, size(input(t)%feature(s)%val, 2)
          do t = 1, this%num_time_steps
             output(:,s) = output(:,s) + &
                  this%transfer%activate( matmul( &
                       this%weight(:,:,t), &
                       input(t)%feature(s)%val(:,v) &
                  ) )
          end do
       end do
    end do

  end function convolutional_get_readout_output

  pure function convolutional_get_readout_differential(this, input) result(output)
    implicit none
    class(convolutional_readout_method_type), intent(in) :: this
    class(state_method_type), dimension(:), intent(in) :: input

    type(feature_type), dimension(this%batch_size) :: output

    integer :: s, v, t

    do s = 1, this%batch_size
       do v = 1, size(input(t)%feature(s)%val, 2)
          do t = 1, this%num_time_steps
             output(s)%val(:,t) = this%transfer%differentiate( &
                  matmul( &
                       this%weight(:,:,t), &
                       input(t)%feature(s)%val(:,v) &
                  ) )
          end do
       end do
    end do

  end function convolutional_get_readout_differential


  pure subroutine convolutional_calculate_readout_partials(this, input, gradient)
    implicit none
    class(convolutional_readout_method_type), intent(inout) :: this
    class(state_method_type), dimension(:), intent(in) :: input
    real(real12), dimension(:,:), intent(in) :: gradient

    integer :: s, v, t
    real(real12), dimension(:), allocatable :: delta

    do concurrent(s=1:this%batch_size)
       !! no message passing transfer function
       
       !! partial derivatives of error wrt weights
       !! dE/dW = o/p(l-1) * delta
       do v = 1, size(input(t)%feature(s)%val, 2)
          do t = 1, this%num_time_steps
  
              delta(:) = gradient(:,s) * this%transfer%differentiate(this%z(s,t)%val(:,v))

              this%dw(:,:,t,s) = this%dw(:,:,t,s) + outer_product(input(t)%feature(s)%val(:,v), delta(:))
              
          end do
          !! SHOULD WORK OUT di FOR EACH TIME STEP
          !! BUT I DON'T KNOW HOW TO HANDLE THAT YET
          !! Well, I get it mathematically, it's just how to include it computationally in a free-form framework 
          delta(:) = gradient(:,s) * this%transfer%differentiate(this%z(s,this%num_time_steps)%val(:,v))
          this%di(s)%val(:,v) = matmul(this%weight(:,:,this%num_time_steps), delta(:))
       end do
    end do
    
  end subroutine convolutional_calculate_readout_partials


end module conv_mpnn_layer
!!!#############################################################################
