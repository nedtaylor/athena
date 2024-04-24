!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module contains implementation of a message passing neural network
!!!#############################################################################
submodule(mpnn_layer) mpnn_layer_submodule
  use constants, only: real12
  use custom_types, only: graph_type
  implicit none
  

contains

  elemental module function feature_add(a, b) result(output)
    class(feature_type), intent(in) :: a, b
    type(feature_type) :: output

    !allocate(output%val(size(a%val,1), size(a%val,2)))
    output%val = a%val + b%val
  end function feature_add

  elemental module function feature_multiply(a, b) result(output)
    class(feature_type), intent(in) :: a, b
    type(feature_type) :: output

    !allocate(output%val(size(a%val,1), size(a%val,2)))
    output%val = a%val * b%val
  end function feature_multiply


!!!#############################################################################
!!! layer setup
!!!#############################################################################
  module function layer_setup( &
       message_method, state_method, readout_method, &
       num_features, num_time_steps, num_outputs, batch_size &
   ) result(layer)
    implicit none
    type(mpnn_layer_type) :: layer
    class(message_method_type), intent(in) :: message_method
    class(state_method_type), intent(in) :: state_method
    class(readout_method_type), intent(in) :: readout_method
    integer, intent(in) :: num_features
    integer, intent(in) :: num_time_steps
    integer, intent(in) :: num_outputs
    integer, optional, intent(in) :: batch_size

    integer :: i

    layer%num_features = num_features
    layer%num_time_steps = num_time_steps
    layer%num_outputs = num_outputs
    if (present(batch_size)) then
       layer%batch_size = batch_size
    else
       layer%batch_size = 1
    end if

    layer%readout = readout_method
    allocate(layer%output(num_outputs, layer%batch_size))

    allocate(layer%message(num_time_steps))
    allocate(layer%state(num_time_steps))
    do i = 1, num_time_steps
       allocate(layer%message(i), source = message_method)
       allocate(layer%state(i), source = state_method)
    end do

  end function layer_setup
!!!#############################################################################


!!!#############################################################################
!!! forward propagation
!!!#############################################################################
  module subroutine forward(this, graph)
    implicit none
    class(mpnn_layer_type), intent(inout) :: this
    type(graph_type), dimension(this%batch_size), intent(in) :: graph

    integer :: v, s, t

    do s = 1, this%batch_size
       do v = 1, graph(s)%num_vertices
          this%state(1)%feature(s)%val(:,v) = graph(s)%vertex(v)%feature
       end do
    end do
    do t = 1, this%num_time_steps
       call this%message(t)%update(this%state(t)%feature, graph)
       call this%state(t)%update(this%message(t+1)%feature, graph)
    end do

    this%output = this%readout%get_output(this%state)

  end subroutine forward
!!!#############################################################################


!!!#############################################################################
!!! backpropagation
!!!#############################################################################
  module subroutine backward(this, graph, gradient)
    implicit none
    class(mpnn_layer_type), intent(inout) :: this
    type(graph_type), dimension(this%batch_size), intent(in) :: graph
    real(real12), dimension( &
         this%readout%num_outputs, &
         this%batch_size &
    ), intent(in) :: gradient

    integer :: s, t

    !df/dv_c = h(M_c) * df/dM_y

    ! M_y = sum_c v_c * h(M_c)     message for output y
    ! h()                          hidden function

    !!! THIS IS THE OUTPUT ERROR, NOT THE INPUT ERROR
    !this%state(this%num_time_steps)%di = &
    !     this%readout%get_differential(this%state, gradient)

    call this%readout%calculate_partials( &
         input = this%state, &
         gradient = gradient &
    )

    call this%state(this%num_time_steps)%calculate_partials( &
         input = this%message(this%num_time_steps)%feature, &
         gradient = this%readout%di, &
         graph = graph &
    )

    do t = this%num_time_steps-1, 2, -1
       !! check if time_step t are all handled correctly here
       call this%message(t+1)%calculate_partials( &
            input = this%state(t)%feature, &
            gradient = this%state(t+1)%di, &
            graph = graph &
       )
       !this%message(t+1)%di = this%state(t+1)%di * &
       !      this%state(t+1)%get_differential( &
       !          this%message(t+1)%feature, graph &
       !      )
       call this%state(t)%calculate_partials( &
            input = this%message(t)%feature, &
            gradient = this%message(t+1)%di, &
            graph = graph &
       )
       !this%state(t)%di = this%message(t+1)%di * &
       !      this%message(t+1)%get_differential( &
       !          this%state(t)%feature, graph &
       !      )

       ! this%di(:,:,t,s) = this%di(:,:,t+1,s) * &
       !       this%state(t+1)%get_differential( &
       !            this%message(t+1)%feature(s)%val(:,:) &
       !       ) * &
       !       this%message(t+1)%get_differential( &
       !            this%state(t)%feature(s)%val(:,:), graph &
       !       )
       
       !! ! this is method dependent
       !! this%dw(:,:,t,s) = this%message(:,t+1,s) * this%v(:,t,s)
    end do
    call this%message(2)%calculate_partials( &
         input = this%state(1)%feature, &
         gradient = this%state(2)%di, &
         graph = graph &
    )

    !do s = 1, this%batch_size
    !   this%di(:,:,s) = this%message(2)%di(s)%val
    !end do


  end subroutine backward
!!!#############################################################################

end submodule mpnn_layer_submodule
!!!#############################################################################