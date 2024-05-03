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

!!!#############################################################################
!!! 
!!!#############################################################################
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

!!!#############################################################################
!!! 
!!!#############################################################################
  module subroutine layer_reduction(this, rhs)
    implicit none
    class(mpnn_layer_type), intent(inout) :: this
    class(learnable_layer_type), intent(in) :: rhs

    !! NOT YET IMPLEMENTED

  end subroutine layer_reduction

  module subroutine layer_merge(this, input)
    implicit none
    class(mpnn_layer_type), intent(inout) :: this
    class(learnable_layer_type), intent(in) :: input

    !! NOT YET IMPLEMENTED

  end subroutine layer_merge
!!!#############################################################################

!!!#############################################################################
!!! 
!!!#############################################################################
  pure module function get_num_params(this) result(num_params)
    implicit none
    class(mpnn_layer_type), intent(in) :: this
    integer :: num_params

    integer :: t

    num_params = 0
    do t = 1, this%method%num_time_steps
       num_params = num_params + this%method%message(t)%get_num_params()
       num_params = num_params + this%method%state(t)%get_num_params()
    end do
    num_params = num_params + this%method%readout%get_num_params()
  end function get_num_params

  pure module function get_params(this) result(params)
    implicit none
    class(mpnn_layer_type), intent(in) :: this
    real(real12), allocatable, dimension(:) :: params
  
    integer :: t

    allocate(params(0))
    do t = 1, this%method%num_time_steps
       params = [ params, this%method%message(t)%get_params() ]
       params = [ params, this%method%state(t)%get_params() ]
    end do
    params = [ params, this%method%readout%get_params() ]
  end function get_params

  pure subroutine set_params(this, params)
    implicit none
    class(mpnn_layer_type), intent(inout) :: this
    real(real12), dimension(:), intent(in) :: params

    integer :: t, istart, iend

    istart = 1
    do t = 1, this%method%num_time_steps
       iend = istart + this%method%message(t)%get_num_params() - 1
       if(iend.gt.istart-1) call this%method%message(t)%set_params(params(istart:iend))
       istart = iend + 1

       iend = istart + this%method%state(t)%get_num_params() - 1
       if(iend.gt.istart-1) call this%method%state(t)%set_params(params(istart:iend))
       istart = iend + 1
    end do
    iend = istart + this%method%readout%get_num_params() - 1
    if(iend.gt.istart-1) call this%method%readout%set_params(params(istart:iend))

  end subroutine set_params

  pure module function get_gradients(this, clip_method) result(gradients)
    implicit none
    class(mpnn_layer_type), intent(in) :: this
    type(clip_type), optional, intent(in) :: clip_method
    real(real12), allocatable, dimension(:) :: gradients

    integer :: t

    allocate(gradients(0))
    if(present(clip_method))then
       do t = 1, this%method%num_time_steps
          gradients = [ gradients, this%method%message(t)%get_gradients(clip_method) ]
          gradients = [ gradients, this%method%state(t)%get_gradients(clip_method) ]
       end do
       gradients = [ gradients, this%method%readout%get_gradients(clip_method) ]
    else
       do t = 1, this%method%num_time_steps
           gradients = [ gradients, this%method%message(t)%get_gradients() ]
           gradients = [ gradients, this%method%state(t)%get_gradients() ]
       end do
       gradients = [ gradients, this%method%readout%get_gradients() ]
    end if
  end function get_gradients

  pure module subroutine set_gradients(this, gradients)
    implicit none
    class(mpnn_layer_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: gradients

    integer :: t

    do t = 1, this%method%num_time_steps
       call this%method%message(t)%set_gradients(gradients)
       call this%method%state(t)%set_gradients(gradients)
    end do
    call this%method%readout%set_gradients(gradients)
  end subroutine set_gradients
!!!#############################################################################


!!!#############################################################################
!!! 
!!!#############################################################################
  pure module function get_method_num_params(this) result(num_params)
    implicit none
    class(base_method_type), intent(in) :: this
    integer :: num_params

    num_params = 0
  end function get_method_num_params

  pure module function get_method_params(this) result(params)
    implicit none
    class(base_method_type), intent(in) :: this
    real(real12), allocatable, dimension(:) :: params

    allocate(params(0))
  end function get_method_params
  
  pure module subroutine set_method_params(this, params)
    implicit none
    class(base_method_type), intent(inout) :: this
    real(real12), dimension(:), intent(in) :: params
    !! nothing to do as no parameters in base method type
  end subroutine set_method_params

  pure module function get_method_gradients(this, clip_method) result(gradients)
    implicit none
    class(base_method_type), intent(in) :: this
    type(clip_type), optional, intent(in) :: clip_method
    real(real12), allocatable, dimension(:) :: gradients

    allocate(gradients(0))
  end function get_method_gradients

  pure module subroutine set_method_gradients(this, gradients)
    implicit none
    class(base_method_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: gradients
    !! nothing to do as no parameters in base method type
  end subroutine set_method_gradients
!!!#############################################################################


!!!#############################################################################
!!! get layer outputs
!!!#############################################################################
  pure subroutine get_output_mpnn(this, output)
    implicit none
    class(mpnn_layer_type), intent(in) :: this
    real(real12), allocatable, dimension(..), intent(out) :: output
  
    select rank(output)
    rank(1)
       output = reshape(this%output, [size(this%output)])
    rank(2)
       output = this%output
    end select
  
  end subroutine get_output_mpnn
!!!#############################################################################


!!!#############################################################################
!!! forward and backward rank procedures
!!!#############################################################################
  pure module subroutine forward_rank(this, input)
    implicit none
    class(mpnn_layer_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: input

    call forward_graph(this, this%graph)
  end subroutine forward_rank
!!!-----------------------------------------------------------------------------
  pure subroutine backward_rank(this, input, gradient)
    implicit none
    class(mpnn_layer_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: input
    real(real12), dimension(..), intent(in) :: gradient

    select rank(gradient); rank(2)
       call backward_graph(this, this%graph, gradient)
    end select
  end subroutine backward_rank
!!!#############################################################################


!!!#############################################################################
!!! layer setup
!!!#############################################################################
  module function layer_setup( &
       method, &
       num_features, num_time_steps, num_outputs, batch_size &
   ) result(layer)
    implicit none
    type(mpnn_layer_type) :: layer
    class(method_container_type), intent(in) :: method
    integer, dimension(2), intent(in) :: num_features
    integer, intent(in) :: num_time_steps
    integer, intent(in) :: num_outputs
    integer, optional, intent(in) :: batch_size

    integer :: i

    layer%output_shape = [ num_outputs ]
    if (present(batch_size)) then
       layer%batch_size = batch_size
    else
       layer%batch_size = 1
    end if

    allocate(layer%method, source=method)

    call layer%init( &
         input_shape = [ num_features(1), num_features(2), num_time_steps ], &
         batch_size = layer%batch_size &
    )

  end function layer_setup
!!!#############################################################################


!!!#############################################################################
!!! layer initialization
!!!#############################################################################
  module subroutine init_mpnn(this, input_shape, batch_size, verbose)
    implicit none
    class(mpnn_layer_type), intent(inout) :: this
    integer, dimension(:), intent(in) :: input_shape
    integer, optional, intent(in) :: batch_size
    integer, optional, intent(in) :: verbose

    this%input_shape = input_shape
    !if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)

    if (present(batch_size)) this%batch_size = batch_size

    !!--------------------------------------------------------------------------
    !! initialise batch size-dependent arrays
    !!--------------------------------------------------------------------------
    if(this%batch_size.gt.0) call this%set_batch_size(this%batch_size)

  end subroutine init_mpnn
!!!#############################################################################


!!!#############################################################################
!!! set batch size
!!!#############################################################################
  subroutine set_batch_size_mpnn(this, batch_size, verbose)
    implicit none
    class(mpnn_layer_type), intent(inout) :: this
    integer, intent(in) :: batch_size
    integer, optional, intent(in) :: verbose
 
    integer :: verbose_ = 0
 
 
    !!--------------------------------------------------------------------------
    !! initialise optional arguments
    !!--------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose
    this%batch_size = batch_size
 
 
    !!--------------------------------------------------------------------------
    !! allocate arrays
    !!--------------------------------------------------------------------------
    if(allocated(this%input_shape))then
       if(allocated(this%output)) deallocate(this%output)
       allocate(this%output(this%output_shape(1), this%batch_size), source=0._real12)
       call this%method%init(this%input_shape, this%output_shape, this%batch_size)
    end if
 
  end subroutine set_batch_size_mpnn
!!!#############################################################################


!!!#############################################################################
!!! set batch size
!!!#############################################################################
  pure module subroutine set_graph(this, graph)
    implicit none
    class(mpnn_layer_type), intent(inout) :: this
    type(graph_type), dimension(this%batch_size), intent(in) :: graph
 
    this%graph = graph 
  end subroutine set_graph
!!!#############################################################################


!!!#############################################################################
!!! forward propagation
!!!#############################################################################
  pure module subroutine forward_graph(this, graph)
    implicit none
    class(mpnn_layer_type), intent(inout) :: this
    type(graph_type), dimension(this%batch_size), intent(in) :: graph

    integer :: v, s, t

    do s = 1, this%batch_size
       do t = 0, this%method%num_time_steps, 1
          if(allocated(this%method%state(t)%feature(s)%val)) deallocate(this%method%state(t)%feature(s)%val)
          allocate(this%method%state(t)%feature(s)%val( &
              this%method%state(t)%num_features, graph(s)%num_vertices &
          ))
          if(t.eq.0) cycle
          if(allocated(this%method%message(t)%feature(s)%val)) deallocate(this%method%message(t)%feature(s)%val)
          allocate(this%method%message(t)%feature(s)%val( &
              this%method%message(t)%num_features, graph(s)%num_vertices &
          ))
       end do
       do v = 1, graph(s)%num_vertices
          this%method%state(1)%feature(s)%val(:,v) = graph(s)%vertex(v)%feature
       end do
    end do
    do t = 1, this%method%num_time_steps
       call this%method%message(t)%update(this%method%state(t-1)%feature, graph)
       call this%method%state(t)%update(this%method%message(t)%feature, graph)
    end do

    call this%method%readout%get_output(this%method%state, this%output)

  end subroutine forward_graph
!!!#############################################################################


!!!#############################################################################
!!! backpropagation
!!!#############################################################################
  pure module subroutine backward_graph(this, graph, gradient)
    implicit none
    class(mpnn_layer_type), intent(inout) :: this
    type(graph_type), dimension(this%batch_size), intent(in) :: graph
    real(real12), dimension( &
         this%output_shape(1), &
         this%batch_size &
    ), intent(in) :: gradient

    integer :: s, t

    !df/dv_c = h(M_c) * df/dM_y

    ! M_y = sum_c v_c * h(M_c)     message for output y
    ! h()                          hidden function

    !!! THIS IS THE OUTPUT ERROR, NOT THE INPUT ERROR
    !this%state(this%num_time_steps)%di = &
    !     this%readout%get_differential(this%state, gradient)

    call this%method%readout%calculate_partials( &
         input = this%method%state, &
         gradient = gradient &
    )

    call this%method%state(this%method%num_time_steps)%calculate_partials( &
         input = this%method%message(this%method%num_time_steps)%feature, &
         gradient = this%method%readout%di, &
         graph = graph &
    )

    do t = this%method%num_time_steps-1, 2, -1
       !! check if time_step t are all handled correctly here
       call this%method%message(t+1)%calculate_partials( &
            input = this%method%state(t)%feature, &
            gradient = this%method%state(t+1)%di, &
            graph = graph &
       )
       !this%message(t+1)%di = this%state(t+1)%di * &
       !      this%state(t+1)%get_differential( &
       !          this%message(t+1)%feature, graph &
       !      )
       call this%method%state(t)%calculate_partials( &
            input = this%method%message(t)%feature, &
            gradient = this%method%message(t+1)%di, &
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
    call this%method%message(2)%calculate_partials( &
         input = this%method%state(1)%feature, &
         gradient = this%method%state(2)%di, &
         graph = graph &
    )

    !do s = 1, this%batch_size
    !   this%di(:,:,s) = this%message(2)%di(s)%val
    !end do


  end subroutine backward_graph
!!!#############################################################################

end submodule mpnn_layer_submodule
!!!#############################################################################