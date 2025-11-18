submodule(athena__base_layer) athena__base_layer_submodule
  !! Submodule containing the implementation of the base layer types
  !!
  !! This submodule contains the implementation of the base layer types
  !! used in the ATHENA library. The base layer types are the abstract
  !! types from which all other layer types are derived. The submodule
  !! contains the implementation of the procedures that are common to
  !! all layer types, such as setting the input shape, getting the
  !! number of parameters, and printing the layer to a file.
  !!
  !! The following procedures are based on code from the neural-fortran library
  !! https://github.com/modern-fortran/neural-fortran/blob/main/src/nf/nf_layer.f90
  !! procedures:
  !! - get_num_params*
  !! - get_params*
  !! - set_params*
  !! - get_gradients*
  !! - set_gradients*
  use coreutils, only: stop_program, print_warning, to_lower, to_upper, icount
  use athena__tools_infile, only: assign_val, assign_vec
  implicit none

contains

!###############################################################################
  module subroutine print_base(this, file, unit, print_header_footer)
    !! Print the layer and wrapping info to a file
    use coreutils, only: to_upper
    implicit none

    ! Arguments
    class(base_layer_type), intent(in) :: this
    !! Instance of the layer
    character(*), optional, intent(in) :: file
    !! File name
    integer, optional, intent(in) :: unit
    !! Unit number
    logical, optional, intent(in) :: print_header_footer
    !! Boolean whether to print header and footer

    ! Local variables
    integer :: unit_
    !! Unit number
    logical :: filename_provided
    !! Boolean whether file is
    logical :: print_header_footer_
    !! Boolean whether to print header and footer


    ! Open file with new unit
    !---------------------------------------------------------------------------
    filename_provided = .false.
    if(present(file).and.present(unit))then
       call stop_program("print_base: both file and unit specified")
    elseif(present(file))then
       filename_provided = .true.
       open(newunit=unit_, file=trim(file), access='append')
    elseif(present(unit))then
       unit_ = unit
    else
       call stop_program("print_base: neither file nor unit specified")
    end if
    print_header_footer_ = .true.
    if(present(print_header_footer)) print_header_footer_ = print_header_footer


    ! Write card
    !---------------------------------------------------------------------------
    if(print_header_footer_) write(unit_,'(A)') to_upper(trim(this%name))
    call this%print_to_unit(unit_)
    if(print_header_footer_) write(unit_,'("END ",A)') to_upper(trim(this%name))


    ! Close unit
    !---------------------------------------------------------------------------
    if(filename_provided) close(unit_)

  end subroutine print_base
!-------------------------------------------------------------------------------
  module subroutine print_to_unit_base(this, unit)
    !! Print the layer to a file
    implicit none

    ! Arguments
    class(base_layer_type), intent(in) :: this
    !! Instance of the layer
    integer, intent(in) :: unit
    !! File unit

    return
  end subroutine print_to_unit_base
!-------------------------------------------------------------------------------
  module subroutine print_to_unit_pool(this, unit)
    !! Print pooling layer to a file
    implicit none

    ! Arguments
    class(pool_layer_type), intent(in) :: this
    !! Instance of the layer
    integer, intent(in) :: unit
    !! File unit

    ! Local variables
    character(100) :: fmt
    !! Format string

    ! Write initial parameters
    !---------------------------------------------------------------------------
    write(fmt,'("(3X,""INPUT_SHAPE ="",",I0,"(1X,I0))")') size(this%input_shape)
    write(unit,fmt) this%input_shape
    if(all(this%pool.eq.this%pool(1)))then
       write(unit,'(3X,"POOL_SIZE =",1X,I0)') this%pool(1)
    else
       write(fmt,'("(3X,""STRIDE ="",",I0,"(1X,I0))")') size(this%pool)
       write(unit,fmt) this%pool
    end if
    if(all(this%strd.eq.this%strd(1)))then
       write(unit,'(3X,"STRIDE =",1X,I0)') this%strd(1)
    else
       write(fmt,'("(3X,""STRIDE ="",",I0,"(1X,I0))")') size(this%strd)
       write(unit,fmt) this%strd
    end if

  end subroutine print_to_unit_pool
!-------------------------------------------------------------------------------
  module subroutine print_to_unit_pad(this, unit)
    !! Print padding layer to a file
    implicit none

    ! Arguments
    class(pad_layer_type), intent(in) :: this
    !! Instance of the layer
    integer, intent(in) :: unit
    !! File unit

    ! Local variables
    character(100) :: fmt
    !! Format string

    ! Write initial parameters
    !---------------------------------------------------------------------------
    write(fmt,'("(3X,""INPUT_SHAPE ="",",I0,"(1X,I0))")') size(this%input_shape)
    write(unit,fmt) this%input_shape
    write(fmt,'("(3X,""PADDING ="",",I0,"(1X,I0))")') size(this%pad)
    write(unit,fmt) this%pad
    write(unit,'(3X,"METHOD = ",A)') trim(this%method)

  end subroutine print_to_unit_pad
!###############################################################################


!###############################################################################
  module subroutine set_rank_base(this, input_rank, output_rank)
    !! Set the input and output ranks of the layer
    implicit none

    ! Arguments
    class(base_layer_type), intent(inout) :: this
    !! Instance of the layer
    integer, intent(in) :: input_rank
    !! Input rank
    integer, intent(in) :: output_rank
    !! Output rank

    !---------------------------------------------------------------------------
    ! Set input and output ranks
    !---------------------------------------------------------------------------
    call stop_program("set_rank_base: this layer cannot have its rank set")

  end subroutine set_rank_base
!###############################################################################


!###############################################################################
  module subroutine set_shape_base(this, input_shape)
    !! Set the input shape of the layer
    implicit none

    ! Arguments
    class(base_layer_type), intent(inout) :: this
    !! Instance of the layer
    integer, dimension(:), intent(in) :: input_shape
    !! Input shape
    character(len=100) :: err_msg
    !! Error message

    !---------------------------------------------------------------------------
    ! initialise input shape
    !---------------------------------------------------------------------------
    if(size(input_shape,dim=1).eq.this%input_rank)then
       this%input_shape = input_shape
    else
       write(err_msg,'("Invalid size of input_shape in ",A,&
            &" expected (",I0,"), got (",I0,")")')  &
            trim(this%name), this%input_rank, size(input_shape,dim=1)
       call stop_program(err_msg)
       return
    end if

  end subroutine set_shape_base
!###############################################################################


!###############################################################################
  module subroutine extract_output_base(this, output)
    !! Get the output of the layer
    implicit none

    ! Arguments
    class(base_layer_type), intent(in) :: this
    !! Instance of the layer
    real(real32), allocatable, dimension(..), intent(out) :: output
    !! Output of the Layer

    if(size(this%output).gt.1)then
       call print_warning("extract_output_base: output has more than one"&
            &" sample, cannot extract")
       return
    end if

    call this%output(1,1)%extract(output)

  end subroutine extract_output_base
!###############################################################################


!###############################################################################
  module subroutine set_ptrs(this)
    !! Set the pointers of the layer
    implicit none

    ! Arguments
    class(base_layer_type), intent(inout), target :: this
    !! Instance of the layer

    ! Local variables
    character(256) :: err_msg
    !! Error message

    ! out_alloc_check: if(allocated(this%output))then
    !    if(this%use_graph_input)then
    !       exit out_alloc_check
    !    elseif(.not.this%output(1,1)%allocated)then
    !       write(err_msg,'("output not allocated for layer ",A," ",I0)') &
    !            trim(this%name), this%id
    !       call stop_program(err_msg)
    !       return
    !    end if
    !    call this%output(1,1)%set_ptr()
    ! end if out_alloc_check

    call this%set_ptrs_hyperparams()

  end subroutine set_ptrs
!-------------------------------------------------------------------------------
  module subroutine set_ptrs_hyperparams(this)
    !! Set the hyperparameter pointers of the layer
    implicit none

    ! Arguments
    class(base_layer_type), intent(inout), target :: this
    !! Instance of the layer

    ! No hyperparameters to set for the base layer
    return
  end subroutine set_ptrs_hyperparams
!###############################################################################


!###############################################################################
  pure module function get_num_params_base(this) result(num_params)
    !! Get the number of parameters in the layer
    implicit none

    ! Arguments
    class(base_layer_type), intent(in) :: this
    !! Instance of the layer
    integer :: num_params
    !! Number of parameters

    ! No parameters in the base layer
    num_params = 0

  end function get_num_params_base
!-------------------------------------------------------------------------------
  pure module function get_num_params_conv(this) result(num_params)
    !! Get the number of parameters in convolutional layer
    implicit none

    ! Arguments
    class(conv_layer_type), intent(in) :: this
    !! Instance of the layer
    integer :: num_params
    !! Number of parameters

    ! num_filters x num_channels x kernel_size + num_biases
    ! num_biases = num_filters
    num_params = this%num_filters * this%num_channels * product(this%knl) + &
         this%num_filters

  end function get_num_params_conv
!-------------------------------------------------------------------------------
  pure module function get_num_params_batch(this) result(num_params)
    !! Get the number of parameters in batch normalisation layer
    implicit none

    ! Arguments
    class(batch_layer_type), intent(in) :: this
    !! Instance of the layer
    integer :: num_params
    !! Number of parameters

    ! num_filters x num_channels x kernel_size + num_biases
    ! num_biases = num_filters
    num_params = 2 * this%num_channels

  end function get_num_params_batch
!###############################################################################


!###############################################################################
  module subroutine forward_derived_base(this, input)
    !! Forward pass for the layer
    implicit none

    ! Arguments
    class(base_layer_type), intent(inout) :: this
    !! Instance of the layer
    class(array_type), dimension(:,:), intent(in) :: input
    !! Input data

    ! Local variables
    integer :: i, j
    !! Loop indices

    do i = 1, size(input, 1)
       do j = 1, size(input, 2)
          if(.not.input(i,j)%allocated)then
             call stop_program('Input to input layer not allocated')
             return
          end if
          this%output(i,j) = input(i,j)
       end do
    end do

  end subroutine forward_derived_base

  module subroutine set_graph_base(this, graph)
    !! Set the graph structure of the input data
    implicit none

    ! Arguments
    class(base_layer_type), intent(inout) :: this
    !! Instance of the layer
    type(graph_type), dimension(:), intent(in) :: graph
    !! Graph structure of input data

    ! Local variables
    integer :: s
    !! Loop index

    if(allocated(this%graph))then
       if(size(this%graph).ne.size(graph))then
          deallocate(this%graph)
          allocate(this%graph(size(graph)))
       end if
    else
       allocate(this%graph(size(graph)))
    end if
    do s = 1, size(graph)
       this%graph(s)%adj_ia = graph(s)%adj_ia
       this%graph(s)%adj_ja = graph(s)%adj_ja
       this%graph(s)%edge_weights = graph(s)%edge_weights
       this%graph(s)%num_edges = graph(s)%num_edges
       this%graph(s)%num_vertices = graph(s)%num_vertices
    end do

    if(this%use_graph_input)then
       if(allocated(this%output))then
          do s = 1, size(graph)
             call this%output(1,s)%allocate( &
                  [ &
                       this%graph(s)%num_vertex_features, &
                       this%graph(s)%num_vertices &
                  ] &
             )
             call this%output(2,s)%allocate( &
                  [ &
                       this%graph(s)%num_edge_features, &
                       this%graph(s)%num_vertices &
                  ] &
             )
          end do
       end if
       call this%set_ptrs()
    end if

  end subroutine set_graph_base

  module subroutine nullify_graph_base(this)
    !! Nullify the forward pass data of the layer to free memory
    implicit none

    ! Arguments
    class(base_layer_type), intent(inout) :: this
    !! Instance of the layer

    ! Local variables
    integer :: i, j
    !! Loop indices

    do i = 1, size(this%output,1)
       do j = 1, size(this%output,2)
          call this%output(i,j)%nullify_graph()
       end do
    end do

  end subroutine nullify_graph_base
!###############################################################################


!###############################################################################
  module subroutine reduce_learnable(this, input)
    !! Merge two learnable layers via summation
    implicit none

    ! Arguments
    class(learnable_layer_type), intent(inout) :: this
    !! Instance of the layer
    class(learnable_layer_type), intent(in) :: input
    !! Instance of a layer

    ! Local variables
    integer :: i
    !! Loop index

    if(allocated(this%params_array).and.allocated(input%params_array))then
       if(size(this%params_array).ne.size(input%params_array))then
          call stop_program("reduce_learnable: incompatible parameter sizes")
          return
       end if
       do i = 1, size(this%params_array,1)
          this%params_array(i) = this%params_array(i) + input%params_array(i)
          if(associated(this%params_array(i)%grad).and.&
               associated(input%params_array(i)%grad))then
             this%params_array(i)%grad = this%params_array(i)%grad + &
                  input%params_array(i)%grad
          end if
       end do
    else
       call stop_program("reduce_learnable: unallocated parameter arrays")
       return
    end if

  end subroutine reduce_learnable
!###############################################################################


!###############################################################################
  module function add_learnable(a, b) result(output)
    !! Add two learnable layers together
    implicit none

    ! Arguments
    class(learnable_layer_type), intent(in) :: a, b
    !! Instances of layers
    class(learnable_layer_type), allocatable :: output
    !! Output layer

    ! Local variables
    integer :: i
    !! Loop index

    output = a
    if(allocated(a%params_array).and.allocated(b%params_array))then
       if(size(a%params_array).ne.size(b%params_array))then
          call stop_program("add_learnable: incompatible parameter sizes")
          return
       end if
       do i = 1, size(a%params_array,1)
          output%params_array(i)%grad => null()
          output%params_array(i) = a%params_array(i) + b%params_array(i)
          if(associated(a%params_array(i)%grad).and.&
               associated(b%params_array(i)%grad))then
             allocate(output%params_array(i)%grad)
             output%params_array(i)%grad = a%params_array(i)%grad + &
                  b%params_array(i)%grad
          end if
       end do
    else
       call stop_program("add_learnable: unallocated parameter arrays")
       return
    end if

  end function add_learnable
!###############################################################################


!###############################################################################
  pure module function get_params(this) result(params)
    !! Get the learnable parameters of the layer
    !!
    !! This function returns the learnable parameters of the layer
    !! as a single array.
    !! This has been modified from the neural-fortran library
    implicit none

    ! Arguments
    class(learnable_layer_type), intent(in) :: this
    !! Instance of the layer
    real(real32), dimension(this%num_params) :: params
    !! Learnable parameters

    ! Local variables
    integer :: i, start_idx, end_idx
    !! Loop indices

    start_idx = 0
    end_idx = 0
    do i = 1, size(this%params_array)
       start_idx = end_idx + 1
       end_idx = start_idx + size(this%params_array(i)%val,1) - 1
       params(start_idx:end_idx) = this%params_array(i)%val(:,1)
    end do

  end function get_params
!###############################################################################


!###############################################################################
  module subroutine set_params(this, params)
    !! Set the learnable parameters of the layer
    !!
    !! This function sets the learnable parameters of the layer
    !! from a single array.
    !! This has been modified from the neural-fortran library
    implicit none

    ! Arguments
    class(learnable_layer_type), intent(inout) :: this
    !! Instance of the layer
    real(real32), dimension(this%num_params), intent(in) :: params
    !! Learnable parameters

    ! Local variables
    integer :: i, start_idx, end_idx
    !! Loop indices

    if(.not.allocated(this%params_array)) then
       call stop_program("set_params: params not allocated")
       return
    end if
    start_idx = 0
    end_idx = 0
    do i = 1, size(this%params_array)
       start_idx = end_idx + 1
       end_idx = start_idx + size(this%params_array(i)%val,1) - 1
       this%params_array(i)%val(:,1) = params(start_idx:end_idx)
    end do

  end subroutine set_params
!###############################################################################


!###############################################################################
  pure module function get_gradients(this, clip_method) result(gradients)
    !! Get the gradients of the layer
    !!
    !! This function returns the gradients of the layer as a single array.
    !! This has been modified from the neural-fortran library
    use athena__clipper, only: clip_type
    implicit none

    ! Arguments
    class(learnable_layer_type), intent(in) :: this
    !! Instance of the layer
    type(clip_type), optional, intent(in) :: clip_method
    !! Method to clip the gradients
    real(real32), dimension(this%num_params) :: gradients
    !! Gradients of the layer

    ! Local variables
    integer :: i, start_idx, end_idx
    !! Loop indices

    if(.not.allocated(this%params_array)) then
       return
    end if
    start_idx = 0
    end_idx = 0
    do i = 1, size(this%params_array)
       start_idx = end_idx + 1
       end_idx = start_idx + size(this%params_array(i)%val,1) - 1
       if(.not.associated(this%params_array(i)%grad)) then
          gradients(start_idx:end_idx) = 0._real32
       else
          gradients(start_idx:end_idx) = this%params_array(i)%grad%val(:,1)
       end if
    end do

    if(present(clip_method)) call clip_method%apply(size(gradients),gradients)

  end function get_gradients
!###############################################################################


!###############################################################################
  module subroutine set_gradients(this, gradients)
    !! Set the gradients of the layer
    !!
    !! This function sets the gradients of the layer from a single array.
    !! This has been modified from the neural-fortran library
    implicit none

    ! Arguments
    class(learnable_layer_type), intent(inout) :: this
    !! Instance of the layer
    real(real32), dimension(..), intent(in) :: gradients
    !! Gradients of the layer

    ! Local variables
    integer :: i, start_idx, end_idx
    !! Loop indices

    start_idx = 0
    end_idx = 0
    select rank(gradients)
    rank(0)
       do i = 1, size(this%params_array)
          if(.not.associated(this%params_array(i)%grad)) then
             this%params_array(i)%grad => this%params_array(i)%create_result()
          end if
          this%params_array(i)%grad%val(:,1) = gradients
       end do
    rank(1)
       do i = 1, size(this%params_array)
          if(.not.associated(this%params_array(i)%grad)) then
             this%params_array(i)%grad => this%params_array(i)%create_result()
          end if
          start_idx = end_idx + 1
          end_idx = start_idx + size(this%params_array(i)%val,1) - 1
          this%params_array(i)%grad%val(:,1) = gradients(start_idx:end_idx)
       end do
    end select

  end subroutine set_gradients
!###############################################################################


!###############################################################################
  module subroutine init_pad(this, input_shape, batch_size, verbose)
    !! Initialise padding layer
    implicit none

    ! Arguments
    class(pad_layer_type), intent(inout) :: this
    !! Instance of the padding layer
    integer, dimension(:), intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: batch_size
    !! Batch size
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: i
    !! Loop index
    integer :: verbose_ = 0
    !! Verbosity level


    !---------------------------------------------------------------------------
    ! Initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose
    if(present(batch_size)) this%batch_size = batch_size


    !---------------------------------------------------------------------------
    ! Initialise input shape
    !---------------------------------------------------------------------------
    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)
    if(.not.allocated(this%orig_bound)) then
       allocate(this%orig_bound(2,this%input_rank-1))
       allocate(this%dest_bound(2,this%input_rank-1))
    end if
    do i = 1, this%input_rank - 1
       this%orig_bound(:,i) = [ 1, this%input_shape(i) ]
       this%dest_bound(:,i) = [ 1, this%input_shape(i) + this%pad(i) * 2 ]
       call this%facets(i)%setup_bounds( &
            length = this%input_shape(:this%input_rank-1), &
            pad = this%pad, &
            imethod = this%imethod &
       )
    end do


    !---------------------------------------------------------------------------
    ! Set up number of channels, width, height
    !---------------------------------------------------------------------------
    this%num_channels = this%input_shape(this%input_rank)
    if(allocated(this%output_shape)) deallocate(this%output_shape)
    allocate( this%output_shape(this%input_rank) )
    this%output_shape(this%input_rank) = this%input_shape(this%input_rank)
    this%output_shape(:this%input_rank-1) = &
         this%input_shape(:this%input_rank-1) + this%pad(:) * 2


    !---------------------------------------------------------------------------
    ! Initialise batch size-dependent arrays
    !---------------------------------------------------------------------------
    if(this%batch_size.gt.0) call this%set_batch_size(this%batch_size)

  end subroutine init_pad
!###############################################################################


!###############################################################################
  module subroutine init_pool(this, input_shape, batch_size, verbose)
    !! Initialise pooling layer
    implicit none

    ! Arguments
    class(pool_layer_type), intent(inout) :: this
    !! Instance of the 1D average pooling layer
    integer, dimension(:), intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: batch_size
    !! Batch size
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level


    !---------------------------------------------------------------------------
    ! Initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose
    if(present(batch_size)) this%batch_size = batch_size


    !---------------------------------------------------------------------------
    ! Initialise input shape
    !---------------------------------------------------------------------------
    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)


    !---------------------------------------------------------------------------
    ! Set up number of channels, width, height
    !---------------------------------------------------------------------------
    this%num_channels = this%input_shape(this%input_rank)
    if(allocated(this%output_shape)) deallocate(this%output_shape)
    allocate( this%output_shape(this%input_rank) )
    this%output_shape(this%input_rank) = this%input_shape(this%input_rank)
    this%output_shape(:this%input_rank-1) = &
         floor( &
              ( &
                   this%input_shape(:this%input_rank-1) - this%pool &
              ) / real(this%strd) &
         ) + 1


    !---------------------------------------------------------------------------
    ! Initialise batch size-dependent arrays
    !---------------------------------------------------------------------------
    if(this%batch_size.gt.0) call this%set_batch_size(this%batch_size)

  end subroutine init_pool
!###############################################################################


!###############################################################################
  module subroutine init_conv(this, input_shape, batch_size, verbose)
    !! Initialise convolutional layer
    use athena__initialiser, only: initialiser_setup
    use athena__misc_types, only: initialiser_type
    implicit none

    ! Arguments
    class(conv_layer_type), intent(inout) :: this
    !! Instance of the layer
    integer, dimension(:), intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: batch_size
    !! Batch size
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level


    !---------------------------------------------------------------------------
    ! initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose
    if(present(batch_size)) this%batch_size = batch_size


    !---------------------------------------------------------------------------
    ! initialise input shape
    !---------------------------------------------------------------------------
    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)


    !---------------------------------------------------------------------------
    ! initialise padding layer, if allocated
    !---------------------------------------------------------------------------
    if(allocated(this%pad_layer)) &
         call this%pad_layer%init(this%input_shape, this%batch_size, verbose_)


    !---------------------------------------------------------------------------
    ! allocate output, activation, bias, and weight shapes
    !---------------------------------------------------------------------------
    ! NOTE: INPUT SHAPE DOES NOT INCLUDE PADDING WIDTH
    ! THIS IS HANDLED AUTOMATICALLY BY THE CODE
    ! ... provide the initial input data shape and let us deal with the padding
    this%num_channels = this%input_shape(this%input_rank)
    if(allocated(this%output_shape)) deallocate(this%output_shape)
    allocate( this%output_shape(this%input_rank) )
    this%output_shape(this%input_rank) = this%num_filters
    this%output_shape(:this%input_rank-1) = floor( &
         ( &
              this%input_shape(:this%input_rank-1) + 2 * this%pad - this%knl &
         ) / real(this%stp) &
    ) + 1
    this%num_params = this%get_num_params()
    allocate(this%weight_shape(this%input_rank + 1,1))
    this%weight_shape(:,1) = [ this%knl, this%num_channels, this%num_filters ]
    this%bias_shape = [this%num_filters]

    if(allocated(this%params_array)) deallocate(this%params_array)
    allocate(this%params_array(2))
    call this%params_array(1)%allocate([this%weight_shape(:,1), 1])
    call this%params_array(1)%set_requires_grad(.true.)
    this%params_array(1)%is_sample_dependent = .false.
    call this%params_array(2)%allocate([this%bias_shape, 1])
    call this%params_array(2)%set_requires_grad(.true.)
    this%params_array(2)%is_sample_dependent = .false.


    !---------------------------------------------------------------------------
    ! initialise weights (kernels)
    !---------------------------------------------------------------------------
    call this%kernel_init%initialise( &
         this%params_array(1)%val(:,1), &
         fan_in = product(this%knl)+1, fan_out = 1, &
         spacing = [ this%knl, this%num_channels, this%num_filters ] &
    )

    ! initialise biases
    !---------------------------------------------------------------------------
    call this%bias_init%initialise( &
         this%params_array(2)%val(:,1), &
         fan_in = product(this%knl)+1, fan_out = 1 &
    )


    !---------------------------------------------------------------------------
    ! initialise batch size-dependent arrays
    !---------------------------------------------------------------------------
    if(this%batch_size.gt.0) call this%set_batch_size(this%batch_size)

  end subroutine init_conv
!###############################################################################


!###############################################################################
  module subroutine init_batch(this, input_shape, batch_size, verbose)
    !! Initialise batch normalisation layer
    use athena__initialiser, only: initialiser_setup
    use athena__misc_types, only: initialiser_type
    implicit none

    ! Arguments
    class(batch_layer_type), intent(inout) :: this
    !! Instance of the layer
    integer, dimension(:), intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: batch_size
    !! Batch size
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    integer :: verbose_ = 0
    class(initialiser_type), allocatable :: t_initialiser


    !---------------------------------------------------------------------------
    ! initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose
    if(present(batch_size)) this%batch_size = batch_size


    !---------------------------------------------------------------------------
    ! initialise input shape
    !---------------------------------------------------------------------------
    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)


    !---------------------------------------------------------------------------
    ! set up number of channels, width, height
    !---------------------------------------------------------------------------
    if(allocated(this%output)) deallocate(this%output)
    allocate(this%output_shape(this%input_rank))
    if(size(this%input_shape).eq.1)then
       this%output_shape(1) = this%input_shape(1)
       this%output_shape(2) = 1
    else
       this%output_shape = this%input_shape
    end if
    this%num_channels = this%input_shape(this%input_rank)
    this%num_params = this%get_num_params()
    allocate(this%params_array(1))
    call this%params_array(1)%allocate([2 * this%num_channels, 1])
    call this%params_array(1)%set_requires_grad(.true.)
    allocate(this%weight_shape(1,1))
    this%weight_shape(:,1) = [ this%num_channels ]
    this%bias_shape = [this%num_channels]


    !---------------------------------------------------------------------------
    ! allocate mean and variance
    !---------------------------------------------------------------------------
    allocate(this%mean(this%num_channels), source=0._real32)
    allocate(this%variance, source=this%mean)


    !---------------------------------------------------------------------------
    ! initialise gamma
    !---------------------------------------------------------------------------
    allocate(t_initialiser, source=initialiser_setup(this%kernel_initialiser))
    t_initialiser%mean = this%gamma_init_mean
    t_initialiser%std  = this%gamma_init_std
    call t_initialiser%initialise(this%params_array(1)%val(1:this%num_channels,1), &
         fan_in =this%num_channels, &
         fan_out=this%num_channels)
    deallocate(t_initialiser)

    ! initialise beta
    !---------------------------------------------------------------------------
    allocate(t_initialiser, source=initialiser_setup(this%bias_initialiser))
    t_initialiser%mean = this%beta_init_mean
    t_initialiser%std  = this%beta_init_std
    call t_initialiser%initialise(this%params_array(1)%val(this%num_channels+1:,1), &
         fan_in =this%num_channels, &
         fan_out=this%num_channels)
    deallocate(t_initialiser)


    !---------------------------------------------------------------------------
    ! initialise moving mean
    !---------------------------------------------------------------------------
    allocate(t_initialiser, &
         source=initialiser_setup(this%moving_mean_initialiser))
    call t_initialiser%initialise(this%mean, &
         fan_in =this%num_channels, &
         fan_out=this%num_channels)
    deallocate(t_initialiser)

    ! initialise moving variance
    !---------------------------------------------------------------------------
    allocate(t_initialiser, &
         source=initialiser_setup(this%moving_variance_initialiser))
    call t_initialiser%initialise(this%variance, &
         fan_in =this%num_channels, &
         fan_out=this%num_channels)
    deallocate(t_initialiser)


    !---------------------------------------------------------------------------
    ! initialise batch size-dependent arrays
    !---------------------------------------------------------------------------
    if(this%batch_size.gt.0) call this%set_batch_size(this%batch_size)

  end subroutine init_batch
!###############################################################################

end submodule athena__base_layer_submodule
