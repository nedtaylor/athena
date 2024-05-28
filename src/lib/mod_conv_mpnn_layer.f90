!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module contains implementation of a ...
!!! ... convolutional message passing neural network layer
!!! Original work by Duvenaud et al. (2015)
!!! https://doi.org/10.48550/arXiv.1509.09292
!!!#############################################################################
module conv_mpnn_layer
  use constants, only: real12
  use misc, only: outer_product
  use custom_types, only: activation_type
  use graph_structure, only: graph_type
  use activation, only: activation_setup
  use clipper, only: clip_type
  use mpnn_layer, only: &
       mpnn_layer_type, method_container_type, &
       message_phase_type, readout_phase_type, &
       feature_type
  implicit none
  

  private
  public :: conv_mpnn_layer_type


!!!-----------------------------------------------------------------------------
!!! convolutional message passing phase
!!!-----------------------------------------------------------------------------
  type, extends(message_phase_type) :: conv_message_phase_type
     !! weight dimensions (feature_out, feature_in, vertex_degree, batch_size)
     real(real12), dimension(:,:,:), allocatable :: weight
     real(real12), dimension(:,:,:,:), allocatable :: dw
     type(feature_type), dimension(:), allocatable :: z
     class(activation_type), allocatable :: transfer
   contains
     procedure :: get_num_params => get_num_params_message_conv
     procedure :: get_params => get_params_message_conv
     procedure :: set_params => set_params_message_conv

     procedure :: get_gradients => get_gradients_message_conv
     procedure :: set_gradients => set_gradients_message_conv

     procedure :: set_shape => set_shape_message_conv

     procedure :: update => update_message_conv
     procedure :: calculate_partials => calculate_partials_message_conv
  end type conv_message_phase_type

  !! interface for the convolutional message passing phase
  !!----------------------------------------------------------------------------
  interface conv_message_phase_type
    module function message_phase_setup( &
         num_vertex_features, num_edge_features, &
         max_vertex_degree, batch_size ) result(message_phase)
      integer, intent(in) :: num_vertex_features, num_edge_features, &
           max_vertex_degree, batch_size
      type(conv_message_phase_type) :: message_phase
    end function message_phase_setup
  end interface conv_message_phase_type


!!!-----------------------------------------------------------------------------
!!! convolutional readout passing phase
!!!-----------------------------------------------------------------------------
  type, extends(readout_phase_type) :: conv_readout_phase_type
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
     procedure :: calculate_partials => calculate_partials_readout_conv
  end type conv_readout_phase_type

  !! interface for the convolutional readout passing phase
  !!----------------------------------------------------------------------------
  interface conv_readout_phase_type
    module function readout_phase_setup( &
         num_time_steps, num_inputs, num_outputs, batch_size ) &
         result(readout_phase)
      integer, intent(in) :: num_time_steps, num_inputs, num_outputs, batch_size
      type(conv_readout_phase_type) :: readout_phase
    end function readout_phase_setup
  end interface conv_readout_phase_type


!!!-----------------------------------------------------------------------------
!!! convolutional MPNN method container
!!!-----------------------------------------------------------------------------
  type, extends(method_container_type) :: conv_method_container_type
    integer :: max_vertex_degree = 6
   contains
    procedure, pass(this) :: init => init_conv_mpnn_method
  end type conv_method_container_type

  !! interface for the convolutional MPNN method container
  !!----------------------------------------------------------------------------
  interface conv_method_container_type
    module function method_setup( &
         num_vertex_features, num_edge_features, num_time_steps, &
         output_shape, &
         max_vertex_degree, &
         batch_size) result(method)
      integer, intent(in) :: num_vertex_features, num_edge_features, &
           num_time_steps
      integer, dimension(1), intent(in) :: output_shape
      integer, intent(in) :: max_vertex_degree, batch_size
      type(conv_method_container_type) :: method
    end function method_setup

  end interface conv_method_container_type


!!!-----------------------------------------------------------------------------
!!! convolutional MPNN layer
!!!-----------------------------------------------------------------------------
  type, extends(mpnn_layer_type) :: conv_mpnn_layer_type
   contains
     procedure :: backward_graph => backward_graph_conv
  end type conv_mpnn_layer_type

  !! interface for the convolutional MPNN layer
  !!----------------------------------------------------------------------------
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

!!!#############################################################################
!!! return number of learnable parameters
!!!#############################################################################
  pure function get_num_params_message_conv(this) result(num_params)
    implicit none
    class(conv_message_phase_type), intent(in) :: this
    integer :: num_params

    num_params = size(this%weight)
  end function get_num_params_message_conv
  !!!-----------------------------------------------------------------------------
  pure function get_num_params_readout_conv(this) result(num_params)
    implicit none
    class(conv_readout_phase_type), intent(in) :: this
    integer :: num_params

    num_params = size(this%weight)
  end function get_num_params_readout_conv
!!!#############################################################################


!!!#############################################################################
!!! return learnable parameters
!!!#############################################################################
  pure module function get_params_message_conv(this) result(params)
    implicit none
    class(conv_message_phase_type), intent(in) :: this
    real(real12), allocatable, dimension(:) :: params
  
    integer :: t

    params = reshape(this%weight, [ size(this%weight) ])
  end function get_params_message_conv
!!!-----------------------------------------------------------------------------
  pure module function get_params_readout_conv(this) result(params)
    implicit none
    class(conv_readout_phase_type), intent(in) :: this
    real(real12), allocatable, dimension(:) :: params
  
    integer :: t

    params = reshape(this%weight, [ size(this%weight) ])
  end function get_params_readout_conv
!!!#############################################################################


!!!#############################################################################
!!! set learnable parameters
!!!#############################################################################
  pure subroutine set_params_message_conv(this, params)
    implicit none
    class(conv_message_phase_type), intent(inout) :: this
    real(real12), dimension(:), intent(in) :: params

    integer :: t

    this%weight = reshape(params, shape(this%weight))
  end subroutine set_params_message_conv
!!!-----------------------------------------------------------------------------
  pure subroutine set_params_readout_conv(this, params)
    implicit none
    class(conv_readout_phase_type), intent(inout) :: this
    real(real12), dimension(:), intent(in) :: params

    integer :: t

    this%weight = reshape(params, shape(this%weight))
  end subroutine set_params_readout_conv
!!!#############################################################################


!!!#############################################################################
!!! return gradients
!!!#############################################################################
  pure function get_gradients_message_conv(this, clip_method) result(gradients)
    implicit none
    class(conv_message_phase_type), intent(in) :: this
    type(clip_type), optional, intent(in) :: clip_method
    real(real12), allocatable, dimension(:) :: gradients

    gradients = reshape(sum(this%dw,dim=4)/this%batch_size, &
         [ size(this%dw,1) * size(this%dw,2) * size(this%dw,3) ])

    if(present(clip_method)) call clip_method%apply(size(gradients),gradients)
  end function get_gradients_message_conv
!!!-----------------------------------------------------------------------------
  pure function get_gradients_readout_conv(this, clip_method) result(gradients)
    implicit none
    class(conv_readout_phase_type), intent(in) :: this
    type(clip_type), optional, intent(in) :: clip_method
    real(real12), allocatable, dimension(:) :: gradients

    gradients = reshape(sum(this%dw,dim=4)/this%batch_size, &
         [ size(this%dw,1) * size(this%dw,2) * size(this%dw,3) ])

    if(present(clip_method)) call clip_method%apply(size(gradients),gradients)
  end function get_gradients_readout_conv
!!!#############################################################################


!!!#############################################################################
!!! set gradients
!!!#############################################################################
  pure subroutine set_gradients_message_conv(this, gradients)
    implicit none
    class(conv_message_phase_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: gradients
  
    select rank(gradients)
    rank(0)
       this%dw = gradients
    rank(1)
       this%dw = spread(reshape(gradients, shape(this%dw(:,:,:,1))), 4, &
            this%batch_size)
    end select

  end subroutine set_gradients_message_conv
!!!-----------------------------------------------------------------------------
  pure subroutine set_gradients_readout_conv(this, gradients)
    implicit none
    class(conv_readout_phase_type), intent(inout) :: this
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
!!! set shape of phases
!!!#############################################################################
  subroutine set_shape_message_conv(this, shape)
    implicit none
    class(conv_message_phase_type), intent(inout) :: this
    integer, dimension(:), intent(in) :: shape

    integer :: s

    
    if(this%use_message)then
       do s = 1, this%batch_size
          if(allocated(this%message(s)%val)) deallocate(this%message(s)%val)
          allocate(this%message(s)%val(this%num_message_features, shape(s)))
       end do
    end if

    do s = 1, this%batch_size
       if(allocated(this%feature(s)%val)) deallocate(this%feature(s)%val)
       allocate(this%feature(s)%val(this%num_outputs, shape(s)))
          
       if(allocated(this%z(s)%val)) deallocate(this%z(s)%val)
       allocate(this%z(s)%val(this%num_outputs, shape(s)))
       if(allocated(this%di(s)%val)) deallocate(this%di(s)%val)
       allocate(this%di(s)%val(this%num_inputs, shape(s)))
    end do

  end subroutine set_shape_message_conv
!!!-----------------------------------------------------------------------------
  subroutine set_shape_readout_conv(this, shape)
    implicit none
    class(conv_readout_phase_type), intent(inout) :: this
    integer, dimension(:), intent(in) :: shape

    integer :: s, t

    do s = 1, this%batch_size
       do t = 0, this%num_time_steps, 1
          if(allocated(this%di(this%batch_size * t + s)%val)) &
               deallocate(this%di(this%batch_size * t + s)%val)
          allocate(this%di(this%batch_size * t + s)%val( &
               this%num_inputs, shape(s) &
          ))

          if(allocated(this%z(t+1,s)%val)) deallocate(this%z(t+1,s)%val)
          allocate(this%z(t+1,s)%val(this%num_outputs, shape(s)))
       end do
    end do

  end subroutine set_shape_readout_conv
!!!#############################################################################


!!!#############################################################################
!!! setup phases
!!!#############################################################################
  module function message_phase_setup( &
         num_vertex_features, num_edge_features, &
         max_vertex_degree, batch_size ) result(message_phase)
    implicit none
    integer, intent(in) :: num_vertex_features, num_edge_features, &
         max_vertex_degree, batch_size
    type(conv_message_phase_type) :: message_phase

    message_phase%num_inputs  = num_vertex_features
    message_phase%num_outputs = num_vertex_features
    message_phase%num_message_features = num_vertex_features + num_edge_features
    message_phase%batch_size  = batch_size
    
    allocate(message_phase%message(batch_size))
    allocate(message_phase%feature(batch_size))
    allocate(message_phase%weight( &
         message_phase%num_message_features, &
         message_phase%num_outputs, &
         max_vertex_degree), source=1._real12)
    allocate(message_phase%dw( &
         message_phase%num_message_features, &
         message_phase%num_outputs, &
         max_vertex_degree, batch_size))
    allocate(message_phase%z(batch_size))
    allocate(message_phase%di(batch_size))
  
    write(*,*) "setting up transfer function"
    allocate(message_phase%transfer, &
         source=activation_setup("sigmoid", 1._real12))
    write(*,*) "transfer function set up"

  end function message_phase_setup
!!!-----------------------------------------------------------------------------
  module function readout_phase_setup( &
         num_time_steps, num_inputs, num_outputs,batch_size &
         ) result(readout_phase)
    implicit none
    integer, intent(in) :: num_time_steps, num_inputs, num_outputs, batch_size
    type(conv_readout_phase_type) :: readout_phase

    readout_phase%num_time_steps = num_time_steps
    readout_phase%num_inputs  = num_inputs
    readout_phase%num_outputs = num_outputs
    readout_phase%batch_size  = batch_size
    allocate(readout_phase%weight( &
         num_inputs, num_outputs, num_time_steps+1), source=1._real12)
    allocate(readout_phase%dw( &
         num_inputs, num_outputs, num_time_steps+1, batch_size))
    allocate(readout_phase%z(num_time_steps+1, batch_size))
    allocate(readout_phase%di(batch_size * (readout_phase%num_time_steps + 1) ))

    write(*,*) "setting up transfer function"
    allocate(readout_phase%transfer, &
         source=activation_setup("softmax", 1._real12))
    write(*,*) "transfer function set up"

  end function readout_phase_setup
!!!#############################################################################


!!!#############################################################################
!!! initialise method container
!!!#############################################################################
  subroutine init_conv_mpnn_method(this, &
       num_vertex_features, num_edge_features, num_time_steps, &
       output_shape, batch_size)
    implicit none
    class(conv_method_container_type), intent(inout) :: this
    integer, intent(in) :: num_vertex_features, num_edge_features, num_time_steps
    integer, dimension(1), intent(in) :: output_shape
    integer, intent(in) :: batch_size


    this%num_features = [num_vertex_features, num_edge_features]
    this%num_time_steps = num_time_steps
    this%num_outputs = output_shape(1)
    if(allocated(this%message)) deallocate(this%message)
    allocate(this%message(0:this%num_time_steps), &
         source = conv_message_phase_type( &
            this%num_features(1), this%num_features(2), &
            this%max_vertex_degree, batch_size &
         ) &
    )
    this%message(0)%use_message = .false.
    if(allocated(this%readout)) deallocate(this%readout)
    allocate(this%readout, &
         source = conv_readout_phase_type( &
              this%num_time_steps, this%num_features(1), &
              this%num_outputs, batch_size &
         ) &
    )

  end subroutine init_conv_mpnn_method
!!!#############################################################################


!!!#############################################################################
!!! setup method container
!!!#############################################################################
  module function method_setup(num_vertex_features, num_edge_features, &
         num_time_steps, output_shape, &
         max_vertex_degree, &
         batch_size) result(method)
    implicit none
    integer, intent(in) :: num_vertex_features, num_edge_features, num_time_steps
    integer, dimension(1), intent(in) :: output_shape
    integer, intent(in) :: max_vertex_degree, batch_size
    type(conv_method_container_type) :: method


    method%num_features = [ num_vertex_features, num_edge_features ]
    method%num_time_steps = num_time_steps
    method%num_outputs = output_shape(1)
    method%max_vertex_degree = max_vertex_degree
    allocate(method%message(0:method%num_time_steps), &
         source = conv_message_phase_type( &
            method%num_features(1), method%num_features(2), &
            method%max_vertex_degree, batch_size &
         ) &
    )
    method%message(0)%use_message = .false.
    allocate(method%readout, &
         source = conv_readout_phase_type( &
              method%num_time_steps, method%num_features(1), &
              method%num_outputs, batch_size &
         ) &
    )

  end function method_setup
!!!#############################################################################


!!!#############################################################################
!!! setup convolutional MPNN
!!!#############################################################################
  module function layer_setup( &
         num_time_steps, &
         num_vertex_features, num_edge_features, &
         num_outputs, &
         max_vertex_degree, &
         batch_size ) result(layer)
    implicit none
    integer, intent(in) :: num_time_steps, num_vertex_features, &
         num_edge_features, num_outputs, max_vertex_degree, batch_size
    type(conv_mpnn_layer_type) :: layer


    layer%batch_size = batch_size
    layer%output_shape = [num_outputs]
    layer%input_shape = [1._real12]

    layer%num_vertex_features = num_vertex_features
    layer%num_edge_features = num_edge_features
    layer%num_time_steps = num_time_steps

    layer%method = conv_method_container_type( &
         num_vertex_features, num_edge_features, num_time_steps, &
         [num_outputs], &
         max_vertex_degree, &
         batch_size &
    )

    allocate(layer%output(num_outputs, layer%batch_size))

  end function layer_setup
!!!#############################################################################


!!!#############################################################################
!!! update procedure for message phase (i.e. forward pass)
!!!#############################################################################
  pure subroutine update_message_conv(this, input, graph)
    implicit none
    class(conv_message_phase_type), intent(inout) :: this
    type(feature_type), dimension(this%batch_size), intent(in) :: input
    type(graph_type), dimension(this%batch_size), intent(in) :: graph

    integer :: s, v, w, e, degree


    if(this%use_message)then
       do concurrent (s = 1: this%batch_size)
          do v = 1, graph(s)%num_vertices
             this%message(s)%val(:,v) = [ &
                  input(s)%val(:,v), &
                  [ ( 0._real12 , w = 1, graph(s)%num_edge_features ) ] ]
             do e = 1, graph(s)%num_edges
                if(any(abs(graph(s)%edge(e)%index).eq.v))then
                   if(graph(s)%edge(e)%index(1) .eq. v)then
                      w = graph(s)%edge(e)%index(2)
                   else
                      w = graph(s)%edge(e)%index(1)
                   end if
                   this%message(s)%val(:,v) = &
                         this%message(s)%val(:,v) + &
                         [ input(s)%val(:,w), graph(s)%edge(e)%feature(:) ]
                end if
             end do
             degree = min(graph(s)%vertex(v)%degree, size(this%weight, 3))
             this%z(s)%val(:,v) = matmul( &
                  this%message(s)%val(:,v), &
                  this%weight(:,:,degree) &
             )
             this%feature(s)%val(:,v) = &
                  this%transfer%activate( this%z(s)%val(:,v) )
          end do
       end do
    else
       do concurrent (s = 1: this%batch_size)
          do v = 1, graph(s)%num_vertices
             degree = min(graph(s)%vertex(v)%degree, size(this%weight, 3))
             this%z(s)%val(:,v) = matmul( &
                  input(s)%val(:,v), &
                  this%weight(:,:,degree) &
             )
             this%feature(s)%val(:,v) = &
                  this%transfer%activate( this%z(s)%val(:,v) )
          end do
       end do
    end if

  end subroutine update_message_conv
!!!#############################################################################


!!!#############################################################################
!!! backward pass for message phase
!!!#############################################################################
  pure subroutine calculate_partials_message_conv(this, input, gradient, graph)
    implicit none
    class(conv_message_phase_type), intent(inout) :: this
    !! hidden features has dimensions (feature, vertex, batch_size)
    type(feature_type), dimension(this%batch_size), intent(in) :: input
    type(feature_type), dimension(this%batch_size), intent(in) :: gradient
    type(graph_type), dimension(this%batch_size), intent(in) :: graph

    integer :: s, v, degree
    real(real12), dimension(:,:), allocatable :: delta


    this%dw = 0._real12
    do concurrent(s=1:this%batch_size)
       !! no message passing transfer function
       delta = gradient(s)%val(:,:) * &
            this%transfer%differentiate(this%z(s)%val(:,:))
       
       !! partial derivatives of error wrt weights
       !! dE/dW = o/p(l-1) * delta
       do v = 1, graph(s)%num_vertices
          degree = min(graph(s)%vertex(v)%degree, size(this%weight, 3))
          !! i.e. outer product of the input and delta
          !! sum weights and biases errors to use in batch gradient descent
          this%dw(:,:,degree,s) = this%dw(:,:,degree,s) + &
               outer_product(input(s)%val(:,v), delta(:,v))
          !! the errors are summed from the delta of the ...
          !! ... 'child' node * 'child' weight
          !! dE/dI(l-1) = sum(weight(l) * delta(l))
          !! this prepares dE/dI for when it is passed into the previous layer
          this%di(s)%val(:this%num_inputs,v) = &
               matmul(this%weight(:this%num_inputs,:,degree), delta(:,v))
       end do
    end do

  end subroutine calculate_partials_message_conv
!!!#############################################################################


!!!#############################################################################
!!! return procedure for readout phase (i.e. forward pass)
!!!#############################################################################
  pure subroutine get_output_readout_conv(this, input, output)
    implicit none
    class(conv_readout_phase_type), intent(inout) :: this
    class(message_phase_type), dimension(0:this%num_time_steps), &
         intent(in) :: input
    real(real12), dimension(this%num_outputs, this%batch_size), &
         intent(out) :: output

    integer :: s, v, t


    do s = 1, this%batch_size
       output(:,s) = 0._real12
       do t = 0, this%num_time_steps, 1
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
!!!#############################################################################


!!!#############################################################################
!!! backward pass for readout phase
!!!#############################################################################
  pure subroutine calculate_partials_readout_conv(this, input, gradient)
    implicit none
    class(conv_readout_phase_type), intent(inout) :: this
    class(message_phase_type), dimension(0:this%num_time_steps), &
         intent(in) :: input
    real(real12), dimension(this%num_outputs, this%batch_size), &
         intent(in) :: gradient

    integer :: s, v, t, num_features
    real(real12), dimension(this%num_outputs) :: delta


    this%dw = 0._real12
    do concurrent(s=1:this%batch_size)
       !! no message passing transfer function
       
       !! partial derivatives of error wrt weights
       !! dE/dW = o/p(l-1) * delta
       do t = 0, this%num_time_steps, 1
          do v = 1, size(input(t)%feature(s)%val, 2)
  
              delta = &
                   gradient(:,s) * &
                   this%transfer%differentiate(this%z(t+1,s)%val(:,v))

              this%dw(:,:,t+1,s) = this%dw(:,:,t+1,s) + &
                   outer_product(input(t)%feature(s)%val(:,v), delta(:))
              
              this%di(this%batch_size * t + s)%val(:,v) = &
                   matmul(this%weight(:,:,t+1), delta(:))
          end do
       end do
    end do
    
  end subroutine calculate_partials_readout_conv
!!!#############################################################################


!!!#############################################################################
!!! backpropagation
!!!#############################################################################
  pure module subroutine backward_graph_conv(this, graph, gradient)
    implicit none
    class(conv_mpnn_layer_type), intent(inout) :: this
    type(graph_type), dimension(this%batch_size), intent(in) :: graph
    real(real12), dimension( &
         this%output_shape(1), &
         this%batch_size &
    ), intent(in) :: gradient

    integer :: s, t


    call this%method%readout%calculate_partials( &
         input = this%method%message, &
         gradient = gradient &
    )

    call this%method%message(this%method%num_time_steps)%calculate_partials( &
         input = this%method%message(this%method%num_time_steps-1)%feature, &
         gradient = this%method%readout%di( &
              this%batch_size * this%num_time_steps + 1 : &
              this%batch_size * (this%num_time_steps + 1) &
         ), &
         graph = graph &
    )

    do t = this%method%num_time_steps - 1, 1, -1
       call this%method%message(t)%calculate_partials( &
            input = this%method%message(t-1)%feature, &
            gradient = &
                 this%method%message(t+1)%di + &
                 this%method%readout%di( &
                      this%batch_size * t + 1 : &
                      this%batch_size * (t + 1) &
                 ), &
            graph = graph &
       )
    end do

  end subroutine backward_graph_conv
!!!#############################################################################

end module conv_mpnn_layer
!!!#############################################################################
