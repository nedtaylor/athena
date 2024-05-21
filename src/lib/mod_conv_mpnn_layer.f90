!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
module conv_mpnn_layer
  use constants, only: real12
  use misc, only: outer_product
  use custom_types, only: graph_type, activation_type
  use activation, only: activation_setup
  use clipper, only: clip_type
  use mpnn_layer, only: &
       mpnn_layer_type, method_container_type, &
       state_method_type, message_method_type, readout_method_type, &
       feature_type
  implicit none
  

  private
  public :: conv_mpnn_layer_type, conv_method_container_type
  public :: conv_readout_method_type



  type, extends(message_method_type) :: conv_message_method_type
   contains
     procedure :: update => update_message_conv
     procedure :: get_differential => get_differential_message_conv
     procedure :: calculate_partials => calculate_partials_message_conv
     procedure :: set_shape => set_shape_message_conv
  end type conv_message_method_type
  interface conv_message_method_type
    module function message_method_setup( &
         num_vertex_features, num_edge_features, batch_size ) result(message_method)
      integer, intent(in) :: num_vertex_features, num_edge_features, batch_size
      type(conv_message_method_type) :: message_method
    end function message_method_setup
  end interface conv_message_method_type


  type, extends(state_method_type) :: conv_state_method_type
     !! weight has dimensions (feature_out, feature_in, vertex_degree, batch_size)
     real(real12), dimension(:,:,:), allocatable :: weight
     real(real12), dimension(:,:,:,:), allocatable :: dw
     type(feature_type), dimension(:), allocatable :: z
     class(activation_type), allocatable :: transfer
   contains
     procedure :: get_num_params => get_num_params_state_conv
     procedure :: get_params => get_params_state_conv
     procedure :: set_params => set_params_state_conv
     procedure :: get_gradients => get_gradients_state_conv
     procedure :: set_gradients => set_gradients_state_conv
     procedure :: set_shape => set_shape_state_conv
     procedure :: update => update_state_conv
     procedure :: get_differential => get_state_differential_conv
     procedure :: calculate_partials => calculate_state_partials_conv
  end type conv_state_method_type
  interface conv_state_method_type
    module function state_method_setup( &
         num_vertex_features, num_edge_features, max_vertex_degree, batch_size ) result(state_method)
      integer, intent(in) :: num_vertex_features, num_edge_features, max_vertex_degree, batch_size
      type(conv_state_method_type) :: state_method
    end function state_method_setup
  end interface conv_state_method_type


  type, extends(readout_method_type) :: conv_readout_method_type
    integer :: num_time_steps
    real(real12), dimension(:,:,:), allocatable :: weight
    real(real12), dimension(:,:,:,:), allocatable :: dw
    type(feature_type), dimension(:,:), allocatable :: z
    class(activation_type), allocatable :: transfer
   contains
     procedure :: get_num_params => get_num_params_readout_conv
     procedure :: get_params => get_params_readout_conv
     procedure :: set_params => set_params_readout_conv
     procedure :: get_gradients => get_gradients_readout_conv
     procedure :: set_gradients => set_gradients_readout_conv
     procedure :: set_shape => set_shape_readout_conv
     procedure :: get_output => get_output_readout_conv
     procedure :: get_differential => get_readout_differential_conv
     procedure :: calculate_partials => calculate_readout_partials_conv
  end type conv_readout_method_type
  interface conv_readout_method_type
    module function readout_method_setup( &
         num_time_steps, num_inputs, num_outputs, batch_size ) result(readout_method)
      integer, intent(in) :: num_time_steps, num_inputs, num_outputs, batch_size
      type(conv_readout_method_type) :: readout_method
    end function readout_method_setup
  end interface conv_readout_method_type




  type, extends(mpnn_layer_type) :: conv_mpnn_layer_type
  end type conv_mpnn_layer_type

  type, extends(method_container_type) :: conv_method_container_type
   contains
    procedure, pass(this) :: init => init_conv_mpnn_method
  end type conv_method_container_type


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



  interface conv_method_container_type
    module function method_setup(input_shape, output_shape, batch_size) result(method)
      integer, dimension(3), intent(in) :: input_shape
      integer, dimension(1), intent(in) :: output_shape
      integer, intent(in) :: batch_size
      type(conv_method_container_type) :: method
    end function method_setup

  end interface conv_method_container_type


contains

!!!#############################################################################
!!! 
!!!#############################################################################
  pure function get_num_params_state_conv(this) result(num_params)
    implicit none
    class(conv_state_method_type), intent(in) :: this
    integer :: num_params

    num_params = size(this%weight)
  end function get_num_params_state_conv
  
  pure function get_num_params_readout_conv(this) result(num_params)
    implicit none
    class(conv_readout_method_type), intent(in) :: this
    integer :: num_params

    num_params = size(this%weight)
  end function get_num_params_readout_conv
!!!#############################################################################


!!!#############################################################################
!!! 
!!!#############################################################################
  pure module function get_params_state_conv(this) result(params)
    implicit none
    class(conv_state_method_type), intent(in) :: this
    real(real12), allocatable, dimension(:) :: params
  
    integer :: t

    params = reshape(this%weight, [ size(this%weight) ])
  end function get_params_state_conv

  pure module function get_params_readout_conv(this) result(params)
    implicit none
    class(conv_readout_method_type), intent(in) :: this
    real(real12), allocatable, dimension(:) :: params
  
    integer :: t

    params = reshape(this%weight, [ size(this%weight) ])
  end function get_params_readout_conv
!!!#############################################################################

!!!#############################################################################
!!! 
!!!#############################################################################
  pure subroutine set_params_state_conv(this, params)
    implicit none
    class(conv_state_method_type), intent(inout) :: this
    real(real12), dimension(:), intent(in) :: params

    integer :: t

    this%weight = reshape(params, shape(this%weight))
  end subroutine set_params_state_conv

  pure subroutine set_params_readout_conv(this, params)
    implicit none
    class(conv_readout_method_type), intent(inout) :: this
    real(real12), dimension(:), intent(in) :: params

    integer :: t

    this%weight = reshape(params, shape(this%weight))
  end subroutine set_params_readout_conv
!!!#############################################################################


!!!#############################################################################
!!! 
!!!#############################################################################
  pure function get_gradients_state_conv(this, clip_method) result(gradients)
    implicit none
    class(conv_state_method_type), intent(in) :: this
    type(clip_type), optional, intent(in) :: clip_method
    real(real12), allocatable, dimension(:) :: gradients

    gradients = reshape(sum(this%dw,dim=4)/this%batch_size, [ size(this%dw,1) * size(this%dw,2) * size(this%dw,3) ])

    if(present(clip_method)) call clip_method%apply(size(gradients),gradients)
  end function get_gradients_state_conv

  pure function get_gradients_readout_conv(this, clip_method) result(gradients)
    implicit none
    class(conv_readout_method_type), intent(in) :: this
    type(clip_type), optional, intent(in) :: clip_method
    real(real12), allocatable, dimension(:) :: gradients

    gradients = reshape(sum(this%dw,dim=4)/this%batch_size, [ size(this%dw,1) * size(this%dw,2) * size(this%dw,3) ])

    if(present(clip_method)) call clip_method%apply(size(gradients),gradients)
  end function get_gradients_readout_conv
!!!#############################################################################


!!!#############################################################################
!!! 
!!!#############################################################################
  pure subroutine set_gradients_state_conv(this, gradients)
    implicit none
    class(conv_state_method_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: gradients
  
    select rank(gradients)
    rank(0)
       this%dw = gradients
    rank(1)
       this%dw = spread(reshape(gradients, shape(this%dw(:,:,:,1))), 4, &
            this%batch_size)
    end select

  end subroutine set_gradients_state_conv

  pure subroutine set_gradients_readout_conv(this, gradients)
    implicit none
    class(conv_readout_method_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: gradients
  
    select rank(gradients)
    rank(0)
       this%dw = gradients
    rank(1)
       this%dw = spread(reshape(gradients, shape(this%dw(:,:,:,1))), 4, &
            this%batch_size)
    end select

  end subroutine set_gradients_readout_conv
!!!#############################################################################

!!!#############################################################################
!!! 
!!!#############################################################################
  subroutine set_shape_message_conv(this, shape)
    implicit none
    class(conv_message_method_type), intent(inout) :: this
    integer, dimension(:), intent(in) :: shape

    integer :: s

    do s = 1, this%batch_size
       if(allocated(this%feature(s)%val)) deallocate(this%feature(s)%val)
       allocate(this%feature(s)%val(this%num_outputs, shape(s)))

       if(allocated(this%di(s)%val)) deallocate(this%di(s)%val)
       allocate(this%di(s)%val(this%num_inputs, shape(s)))
    end do

  end subroutine set_shape_message_conv


  subroutine set_shape_state_conv(this, shape)
    implicit none
    class(conv_state_method_type), intent(inout) :: this
    integer, dimension(:), intent(in) :: shape

    integer :: s

    do s = 1, this%batch_size
       if(allocated(this%feature(s)%val)) deallocate(this%feature(s)%val)
       allocate(this%feature(s)%val(this%num_outputs, shape(s)))
          
       if(allocated(this%z(s)%val)) deallocate(this%z(s)%val)
       allocate(this%z(s)%val(this%num_outputs, shape(s)))
       if(allocated(this%di(s)%val)) deallocate(this%di(s)%val)
       allocate(this%di(s)%val(this%num_inputs, shape(s)))
    end do

  end subroutine set_shape_state_conv

  subroutine set_shape_readout_conv(this, shape)
    implicit none
    class(conv_readout_method_type), intent(inout) :: this
    integer, dimension(:), intent(in) :: shape

    integer :: s, t

    do t = 0, this%num_time_steps, 1
       do s = 1, this%batch_size
          if(allocated(this%di(s)%val)) deallocate(this%di(s)%val)
          allocate(this%di(s)%val(this%num_inputs, shape(s)))

          if(allocated(this%z(t+1,s)%val)) deallocate(this%z(t+1,s)%val)
          allocate(this%z(t+1,s)%val(this%num_outputs, shape(s)))
       end do
    end do

  end subroutine set_shape_readout_conv
!!!#############################################################################


  module function message_method_setup( &
         num_vertex_features, num_edge_features, batch_size ) result(message_method)
    implicit none
    integer, intent(in) :: num_vertex_features, num_edge_features, batch_size
    type(conv_message_method_type) :: message_method

    message_method%num_inputs  = num_vertex_features
    message_method%num_outputs = num_vertex_features + num_edge_features
    message_method%batch_size  = batch_size
    
    allocate(message_method%feature(batch_size))
    allocate(message_method%di(batch_size))

  end function message_method_setup

  module function state_method_setup( &
         num_vertex_features, num_edge_features, max_vertex_degree, batch_size ) result(state_method)
    implicit none
    integer, intent(in) :: num_vertex_features, num_edge_features, batch_size, max_vertex_degree
    type(conv_state_method_type) :: state_method

    state_method%num_inputs  = num_vertex_features + num_edge_features
    state_method%num_outputs = num_vertex_features
    state_method%batch_size  = batch_size

    !!! MAXIMUM VERTEX DEGREE
    allocate(state_method%feature(batch_size))
    allocate(state_method%weight(state_method%num_inputs, state_method%num_outputs, max_vertex_degree))
    allocate(state_method%dw(state_method%num_inputs, state_method%num_outputs, max_vertex_degree, batch_size))
    allocate(state_method%z(batch_size))
    allocate(state_method%di(batch_size))
  
    write(*,*) "setting up transfer function"
    allocate(state_method%transfer, &
         source=activation_setup("sigmoid", 1._real12))
    write(*,*) "transfer function set up"

  end function state_method_setup

  module function readout_method_setup( &
         num_time_steps, num_inputs, num_outputs,batch_size ) result(readout_method)
    implicit none
    integer, intent(in) :: num_time_steps, num_inputs, num_outputs, batch_size
    type(conv_readout_method_type) :: readout_method

    readout_method%num_time_steps = num_time_steps
    readout_method%num_inputs  = num_inputs
    readout_method%num_outputs = num_outputs
    readout_method%batch_size  = batch_size
    allocate(readout_method%weight(num_inputs, num_outputs, num_time_steps+1))
    allocate(readout_method%dw(num_inputs, num_outputs, num_time_steps+1, batch_size))
    allocate(readout_method%z(num_time_steps+1, batch_size))
    allocate(readout_method%di(batch_size))

    write(*,*) "setting up transfer function"
    allocate(readout_method%transfer, &
         source=activation_setup("softmax", 1._real12))
    write(*,*) "transfer function set up"

  end function readout_method_setup



  subroutine init_conv_mpnn_method(this, input_shape, output_shape, batch_size)
    implicit none
    class(conv_method_container_type), intent(inout) :: this
    integer, dimension(3), intent(in) :: input_shape
    integer, dimension(1), intent(in) :: output_shape
    integer, intent(in) :: batch_size


    this%num_features = input_shape(:2)
    this%num_time_steps = input_shape(3)
    this%num_outputs = output_shape(1)
    if(allocated(this%message)) deallocate(this%message)
    allocate(this%message(this%num_time_steps), &
         source = conv_message_method_type( &
            this%num_features(1), this%num_features(2), batch_size &
         ) &
    )
    if(allocated(this%state)) deallocate(this%state)
    allocate(this%state(0:this%num_time_steps), &
         source = conv_state_method_type( &
            this%num_features(1), this%num_features(2), 6, batch_size &
         ) &
    )
    if(allocated(this%readout)) deallocate(this%readout)
    allocate(this%readout, &
         source = conv_readout_method_type( &
              this%num_time_steps, this%num_features(1), this%num_outputs, batch_size &
         ) &
    )

  end subroutine init_conv_mpnn_method


  module function method_setup(input_shape, output_shape, batch_size) result(method)
    implicit none
    integer, dimension(3), intent(in) :: input_shape
    integer, dimension(1), intent(in) :: output_shape
    integer, intent(in) :: batch_size
    type(conv_method_container_type) :: method


    method%num_features = input_shape(:2)
    method%num_time_steps = input_shape(3)
    method%num_outputs = output_shape(1)
    allocate(method%message(method%num_time_steps), &
         source = conv_message_method_type( &
            method%num_features(1), method%num_features(2), batch_size &
         ) &
    )
    write(*,*) "num_messages", size(method%message)
    allocate(method%state(0:method%num_time_steps), &
         source = conv_state_method_type( &
            method%num_features(1), method%num_features(2), 4, batch_size &
         ) &
    )
    write(*,*) "num_states", size(method%state)
    allocate(method%readout, &
         source = conv_readout_method_type( &
              method%num_time_steps, method%num_features(1), method%num_outputs, batch_size &
         ) &
    )

  end function method_setup



  module function layer_setup( &
         num_time_steps, &
         num_vertex_features, num_edge_features, &
         num_outputs, batch_size ) result(layer)
    implicit none
    integer, intent(in) :: num_time_steps, num_vertex_features, &
         num_edge_features, num_outputs, batch_size
    type(conv_mpnn_layer_type) :: layer

    layer%batch_size = batch_size
    layer%output_shape = [num_outputs]
    layer%input_shape = [num_vertex_features, num_edge_features, num_time_steps] !!! MAY CHANGE THIS, provide MPNN with extra data it can pass through


    layer%method = conv_method_container_type( &
         [num_vertex_features, num_edge_features, num_time_steps], [num_outputs], batch_size &
    )
    !call layer%method%init(&
    !     [num_vertex_features, num_edge_features, num_time_steps], [num_outputs], batch_size)

    allocate(layer%output(num_outputs, layer%batch_size))

  end function layer_setup


  pure subroutine update_message_conv(this, input, graph)
    implicit none
    class(conv_message_method_type), intent(inout) :: this
    type(feature_type), dimension(this%batch_size), intent(in) :: input
    type(graph_type), dimension(this%batch_size), intent(in) :: graph

    integer :: s, v, w

    do s = 1, this%batch_size
       do v = 1, graph(s)%num_vertices
         this%feature(s)%val(:,v) = 0._real12
         do w = 1, graph(s)%num_vertices
             if(graph(s)%adjacency(v,w) .ne. 0) then
               this%feature(s)%val(:,v) = &
                     this%feature(s)%val(:,v) + &
                     [ input(s)%val(:,w), graph(s)%edge(abs(graph(s)%adjacency(v,w)))%feature(:) ]
             end if
         end do
       end do
    end do

  end subroutine update_message_conv

  pure function get_differential_message_conv(this, input, graph) &
       result(output)
    implicit none
    class(conv_message_method_type), intent(in) :: this
    type(feature_type), dimension(this%batch_size), intent(in) :: input
    type(graph_type), dimension(this%batch_size), intent(in) :: graph
    
    type(feature_type), dimension(this%batch_size) :: output

    integer :: s, v

    do s = 1, this%batch_size
       do v = 1, graph(s)%num_vertices
          output(s)%val(:,v) = 1._real12
       end do
    end do

  end function get_differential_message_conv

  pure subroutine calculate_partials_message_conv(this, input, gradient, graph)
    implicit none
    class(conv_message_method_type), intent(inout) :: this
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
       this%di(s)%val(:,:) = gradient(s)%val(:graph(s)%num_vertex_features,:)
    end do

  end subroutine calculate_partials_message_conv


  pure subroutine update_state_conv(this, input, graph)
    implicit none
    class(conv_state_method_type), intent(inout) :: this
    type(feature_type), dimension(this%batch_size), intent(in) :: input
    type(graph_type), dimension(this%batch_size), intent(in) :: graph

    integer :: s, v

    do s = 1, this%batch_size
       ! if(allocated(this%z(s)%val)) deallocate(this%z(s)%val)
       ! allocate(this%z(s)%val(size(input(s)%val, 1), graph(s)%num_vertices))
       do v = 1, graph(s)%num_vertices
          this%z(s)%val(:,v) = matmul( &
               input(s)%val(:,v), &
               this%weight(:,:,graph(s)%vertex(v)%degree) &
          )
          this%feature(s)%val(:,v) = this%transfer%activate( this%z(s)%val(:,v))
       end do
    end do

  end subroutine update_state_conv

  pure function get_state_differential_conv(this, input, graph) result(output)
    implicit none
    class(conv_state_method_type), intent(in) :: this
    type(feature_type), dimension(this%batch_size), intent(in) :: input
    type(graph_type), dimension(this%batch_size), intent(in) :: graph

    type(feature_type), dimension(this%batch_size):: output

    integer :: v, s

    do s = 1, this%batch_size
       do v = 1, graph(s)%num_vertices
          output(s)%val(:,v) = this%transfer%differentiate(this%z(s)%val(:,v))
       end do
    end do

  end function get_state_differential_conv

  pure subroutine calculate_state_partials_conv(this, input, gradient, graph)
    implicit none
    class(conv_state_method_type), intent(inout) :: this
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
       delta = gradient(s)%val(:,:) * &
            this%transfer%differentiate(this%z(s)%val(:,:))
       if(allocated(this%di(s)%val)) deallocate(this%di(s)%val)
       allocate(this%di(s)%val(size(input(s)%val, 1), &
           size(input(s)%val, 2)))
       
       !! partial derivatives of error wrt weights
       !! dE/dW = o/p(l-1) * delta
       do v = 1, graph(s)%num_vertices
          degree = min(graph(s)%vertex(v)%degree, size(this%weight, 3))
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

  end subroutine calculate_state_partials_conv


  pure subroutine get_output_readout_conv(this, input, output)
    implicit none
    class(conv_readout_method_type), intent(inout) :: this
    class(state_method_type), dimension(0:this%num_time_steps), intent(in) :: input
    real(real12), dimension(this%num_outputs, this%batch_size), intent(out) :: output

    integer :: s, v, t

    do s = 1, this%batch_size
       output(:,s) = 0._real12
       do t = 0, this%num_time_steps, 1
          ! if(allocated(this%z(t+1,s)%val)) deallocate(this%z(t+1,s)%val)
          ! allocate(this%z(t+1,s)%val(this%num_outputs, size(input(t)%feature(s)%val, 2)))
          do v = 1, size(input(t)%feature(s)%val, 2)
             this%z(t+1,s)%val(:,v) = matmul( &
                  input(t)%feature(s)%val(:,v), &
                  this%weight(:,:,t+1) &
             )
             output(:,s) = output(:,s) + &
                  this%transfer%activate( this%z(t+1,s)%val(:,v) )
          end do
       end do
    end do

  end subroutine get_output_readout_conv

  pure function get_readout_differential_conv(this, input) result(output)
    implicit none
    class(conv_readout_method_type), intent(in) :: this
    class(state_method_type), dimension(0:this%num_time_steps), intent(in) :: input

    type(feature_type), dimension(this%batch_size) :: output

    integer :: s, v, t

    do s = 1, this%batch_size
       do t = 0, this%num_time_steps
          do v = 1, size(input(t)%feature(s)%val, 2)
             output(s)%val(:,t) = this%transfer%differentiate(this%z(t,s)%val(:,v))
          end do
       end do
    end do

  end function get_readout_differential_conv


  pure subroutine calculate_readout_partials_conv(this, input, gradient)
    implicit none
    class(conv_readout_method_type), intent(inout) :: this
    class(state_method_type), dimension(0:this%num_time_steps), intent(in) :: input
    real(real12), dimension(this%num_outputs, this%batch_size), intent(in) :: gradient

    integer :: s, v, t
    real(real12), dimension(this%num_outputs) :: delta

    do concurrent(s=1:this%batch_size)
       !! no message passing transfer function
       
       !! partial derivatives of error wrt weights
       !! dE/dW = o/p(l-1) * delta
       do t = 0, this%num_time_steps
          if(allocated(this%di(s)%val)) deallocate(this%di(s)%val)
           allocate(this%di(s)%val(size(input(this%num_time_steps)%feature(s)%val, 1), &
                size(input(this%num_time_steps)%feature(s)%val, 2)))
          do v = 1, size(input(t)%feature(s)%val, 2)
  
              delta = gradient(:,s) * this%transfer%differentiate(this%z(t+1,s)%val(:,v))

              this%dw(:,:,t+1,s) = this%dw(:,:,t+1,s) + outer_product(input(t)%feature(s)%val(:,v), delta(:))

              delta(:) = gradient(:,s) * this%transfer%differentiate(this%z(this%num_time_steps+1,s)%val(:,v))
              
              if(t .ne. this%num_time_steps) cycle
              this%di(s)%val(:,v) = matmul(this%weight(:,:,this%num_time_steps+1), delta(:))
          end do
          !! SHOULD WORK OUT di FOR EACH TIME STEP
          !! BUT I DON'T KNOW HOW TO HANDLE THAT YET
          !! Well, I get it mathematically, it's just how to include it computationally in a free-form framework
          !this%di(t,s)%val(:,v) = matmul(this%weight(:,:,this%num_time_steps+1), delta(:))
       end do
    end do
    
  end subroutine calculate_readout_partials_conv

end module conv_mpnn_layer
!!!#############################################################################
