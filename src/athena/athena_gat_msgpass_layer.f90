module athena__gat_msgpass_layer
  !! Module implementing Graph Attention Networks (GAT)
  !!
  !! This module implements the graph attention layer from Velickovic et al.
  !! (2018) with multi-head attention for learning node representations.
  !!
  !! Mathematical operation per head k:
  !! \[ e_{ij}^{(k)} = \text{LeakyReLU}\left( \mathbf{a}^{(k)T} [\mathbf{W}^{(k)} \mathbf{h}_i \| \mathbf{W}^{(k)} \mathbf{h}_j] \right) \]
  !! \[ \alpha_{ij}^{(k)} = \text{softmax}_j(e_{ij}^{(k)}) = \frac{\exp(e_{ij}^{(k)})}{\sum_{m \in \mathcal{N}(i)} \exp(e_{im}^{(k)})} \]
  !! \[ \mathbf{h}_i'^{(k)} = \sigma\left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)} \mathbf{W}^{(k)} \mathbf{h}_j \right) \]
  !!
  !! Multi-head output (concatenation):
  !! \[ \mathbf{h}_i' = \|_{k=1}^{K} \mathbf{h}_i'^{(k)} \]
  !!
  !! Multi-head output (averaging, for final layer):
  !! \[ \mathbf{h}_i' = \sigma\left( \frac{1}{K} \sum_{k=1}^{K} \mathbf{h}_i'^{(k)} \right) \]
  !!
  !! where:
  !! * \( \mathbf{W}^{(k)} \in \mathbb{R}^{F'/K \times F} \) is a learnable weight matrix
  !! * \( \mathbf{a}^{(k)} \in \mathbb{R}^{2F'/K} \) is the attention vector (split as left/right)
  !! * \( \alpha_{ij} \) are normalised attention coefficients
  !! * \( K \) is the number of attention heads
  !! * \( \sigma \) is the activation function
  !!
  !! Reference: Velickovic et al. (2018), Graph Attention Networks, ICLR
  use coreutils, only: real32, stop_program
  use graphstruc, only: graph_type
  use athena__misc_types, only: base_actv_type, base_init_type
  use diffstruc, only: array_type
  use athena__base_layer, only: base_layer_type
  use athena__msgpass_layer, only: msgpass_layer_type
  use athena__diffstruc_extd, only: gat_propagate
  use diffstruc, only: matmul
  implicit none


  private

  public :: gat_msgpass_layer_type
  public :: read_gat_msgpass_layer


!-------------------------------------------------------------------------------
! GAT message passing layer
!-------------------------------------------------------------------------------
  type, extends(msgpass_layer_type) :: gat_msgpass_layer_type
     integer :: num_heads = 1
     !! Number of attention heads
     logical :: concat_heads = .true.
     !! Whether to concatenate heads (.true.) or average them (.false.)
     real(real32) :: negative_slope = 0.2_real32
     !! Negative slope for LeakyReLU in attention

     !! Parameters layout (2 entries per time step):
     !! params(2*t-1) stores weight matrix W for time step t:
     !!   val(F_out_per_head * num_heads * F_in, 1)
     !! params(2*t) stores attention vectors for time step t:
     !!   val(F_out_per_head * 2 * num_heads, 1)
     !!   Layout: [a_l^1, ..., a_l^K, a_r^1, ..., a_r^K]

   contains
     procedure, pass(this) :: get_num_params => get_num_params_gat
     procedure, pass(this) :: set_hyperparams => set_hyperparams_gat
     procedure, pass(this) :: init => init_gat
     procedure, pass(this) :: print_to_unit => print_to_unit_gat
     procedure, pass(this) :: read => read_gat
     procedure, pass(this) :: update_message => update_message_gat
     procedure, pass(this) :: update_readout => update_readout_gat
  end type gat_msgpass_layer_type

  interface gat_msgpass_layer_type
     module function layer_setup( &
          num_vertex_features, num_time_steps, &
          num_heads, concat_heads, negative_slope, &
          activation, &
          kernel_initialiser, &
          verbose &
     ) result(layer)
       integer, dimension(:), intent(in) :: num_vertex_features
       integer, intent(in) :: num_time_steps
       integer, optional, intent(in) :: num_heads
       logical, optional, intent(in) :: concat_heads
       real(real32), optional, intent(in) :: negative_slope
       class(*), optional, intent(in) :: activation, kernel_initialiser
       integer, optional, intent(in) :: verbose
       type(gat_msgpass_layer_type) :: layer
     end function layer_setup
  end interface gat_msgpass_layer_type

contains


!###############################################################################
  pure function get_num_params_gat(this) result(num_params)
    !! Get total number of learnable parameters
    !! Includes weight matrices W and attention vectors a for all heads/time steps
    implicit none
    class(gat_msgpass_layer_type), intent(in) :: this
    integer :: num_params
    integer :: t, f_out_per_head

    num_params = 0
    do t = 1, this%num_time_steps
       if(this%concat_heads) then
          f_out_per_head = this%num_vertex_features(t) / this%num_heads
       else
          f_out_per_head = this%num_vertex_features(t)
       end if
       ! Weight params: F_out_total * F_in (where F_out_total = f_out_per_head * num_heads)
       num_params = num_params + &
            f_out_per_head * this%num_heads * this%num_vertex_features(t-1)
       ! Attention params: f_out_per_head * 2 * num_heads (left + right per head)
       num_params = num_params + f_out_per_head * 2 * this%num_heads
    end do
  end function get_num_params_gat
!###############################################################################


!###############################################################################
  module function layer_setup( &
       num_vertex_features, num_time_steps, &
       num_heads, concat_heads, negative_slope, &
       activation, &
       kernel_initialiser, &
       verbose &
  ) result(layer)
    use athena__activation, only: activation_setup
    use athena__initialiser, only: initialiser_setup
    implicit none

    integer, dimension(:), intent(in) :: num_vertex_features
    integer, intent(in) :: num_time_steps
    integer, optional, intent(in) :: num_heads
    logical, optional, intent(in) :: concat_heads
    real(real32), optional, intent(in) :: negative_slope
    class(*), optional, intent(in) :: activation, kernel_initialiser
    integer, optional, intent(in) :: verbose
    type(gat_msgpass_layer_type) :: layer

    integer :: verbose_ = 0
    class(base_actv_type), allocatable :: activation_
    class(base_init_type), allocatable :: kernel_initialiser_

    if(present(verbose)) verbose_ = verbose

    if(present(activation))then
       activation_ = activation_setup(activation)
    else
       activation_ = activation_setup("none")
    end if

    if(present(kernel_initialiser))then
       kernel_initialiser_ = initialiser_setup(kernel_initialiser)
    end if

    if(present(num_heads)) layer%num_heads = num_heads
    if(present(concat_heads)) layer%concat_heads = concat_heads
    if(present(negative_slope)) layer%negative_slope = negative_slope

    call layer%set_hyperparams( &
         num_vertex_features = num_vertex_features, &
         num_time_steps = num_time_steps, &
         activation = activation_, &
         kernel_initialiser = kernel_initialiser_, &
         verbose = verbose_ &
    )

    call layer%init(input_shape=[layer%num_vertex_features(0), 0])

  end function layer_setup
!###############################################################################


!###############################################################################
  subroutine set_hyperparams_gat( &
       this, &
       num_vertex_features, &
       num_time_steps, &
       activation, &
       kernel_initialiser, &
       verbose &
  )
    use athena__activation, only: activation_setup
    use athena__initialiser, only: get_default_initialiser, initialiser_setup
    implicit none

    class(gat_msgpass_layer_type), intent(inout) :: this
    integer, dimension(:), intent(in) :: num_vertex_features
    integer, intent(in) :: num_time_steps
    class(base_actv_type), allocatable, intent(in) :: activation
    class(base_init_type), allocatable, intent(in) :: kernel_initialiser
    integer, optional, intent(in) :: verbose

    integer :: t, f_out_per_head
    character(len=256) :: buffer

    this%name = 'gat'
    this%type = 'msgp'
    this%input_rank = 2
    this%output_rank = 2
    this%use_graph_output = .true.
    this%num_time_steps = num_time_steps

    if(allocated(this%num_vertex_features)) &
         deallocate(this%num_vertex_features)
    if(allocated(this%num_edge_features)) &
         deallocate(this%num_edge_features)
    if(size(num_vertex_features, 1) .eq. 1) then
       allocate( &
            this%num_vertex_features(0:num_time_steps), &
            source = num_vertex_features(1) &
       )
    elseif(size(num_vertex_features, 1) .eq. num_time_steps + 1) then
       allocate( &
            this%num_vertex_features(0:this%num_time_steps), &
            source = num_vertex_features &
       )
    else
       call stop_program( &
            "Error: num_vertex_features must be a scalar or a vector of length &
            &num_time_steps + 1" &
       )
    end if
    allocate( this%num_edge_features(0:this%num_time_steps), source = 0 )
    this%use_graph_input = .true.

    if(allocated(this%activation)) deallocate(this%activation)
    if(.not.allocated(activation))then
       this%activation = activation_setup("none")
    else
       allocate(this%activation, source=activation)
    end if
    if(allocated(this%kernel_init)) deallocate(this%kernel_init)
    if(.not.allocated(kernel_initialiser))then
       buffer = get_default_initialiser(this%activation%name)
       this%kernel_init = initialiser_setup(buffer)
    else
       allocate(this%kernel_init, source=kernel_initialiser)
    end if

    if(present(verbose))then
       if(abs(verbose).gt.0)then
          write(*,'("GAT activation function: ",A)') &
               trim(this%activation%name)
          write(*,'("GAT kernel initialiser: ",A)') &
               trim(this%kernel_init%name)
          write(*,'("GAT num_heads: ",I0)') this%num_heads
          write(*,'("GAT concat_heads: ",L1)') this%concat_heads
       end if
    end if

    if(allocated(this%num_params_msg)) deallocate(this%num_params_msg)
    allocate(this%num_params_msg(1:this%num_time_steps))
    do t = 1, this%num_time_steps
       if(this%concat_heads) then
          f_out_per_head = this%num_vertex_features(t) / this%num_heads
       else
          f_out_per_head = this%num_vertex_features(t)
       end if
       ! Weight params + attention params
       this%num_params_msg(t) = &
            f_out_per_head * this%num_heads * this%num_vertex_features(t-1) + &
            f_out_per_head * 2 * this%num_heads
    end do
    if(allocated(this%input_shape)) deallocate(this%input_shape)
    if(allocated(this%output_shape)) deallocate(this%output_shape)

  end subroutine set_hyperparams_gat
!###############################################################################


!###############################################################################
  subroutine init_gat(this, input_shape, verbose)
    use athena__initialiser, only: initialiser_setup
    implicit none

    class(gat_msgpass_layer_type), intent(inout) :: this
    integer, dimension(:), intent(in) :: input_shape
    integer, optional, intent(in) :: verbose

    integer :: t, k, f_out_per_head
    integer :: verbose_ = 0

    if(present(verbose)) verbose_ = verbose

    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)
    this%output_shape = [this%num_vertex_features(this%num_time_steps), 0]
    this%num_params = this%get_num_params()

    if(allocated(this%weight_shape)) deallocate(this%weight_shape)
    allocate(this%weight_shape(2, 2*this%num_time_steps))

    ! Allocate params: 2 entries per time step (weight + attention)
    if(allocated(this%params)) deallocate(this%params)
    allocate(this%params(2 * this%num_time_steps))

    do t = 1, this%num_time_steps
       if(this%concat_heads) then
          f_out_per_head = this%num_vertex_features(t) / this%num_heads
       else
          f_out_per_head = this%num_vertex_features(t)
       end if

       this%weight_shape(:, 2*t-1) = &
            [ f_out_per_head * this%num_heads, this%num_vertex_features(t-1) ]
       this%weight_shape(:, 2*t) = &
            [ f_out_per_head * 2 * this%num_heads, 1 ]

       ! W: params(2*t-1), val(F_out_total * F_in, 1)
       call this%params(2*t-1)%allocate( &
            array_shape = [ &
                 f_out_per_head * this%num_heads, &
                 this%num_vertex_features(t-1), 1 ] &
       )
       call this%params(2*t-1)%set_requires_grad(.true.)
       this%params(2*t-1)%is_sample_dependent = .false.
       this%params(2*t-1)%is_temporary = .false.
       this%params(2*t-1)%fix_pointer = .true.

       ! Attention: params(2*t), val(f_per_head * 2 * num_heads, 1)
       call this%params(2*t)%allocate( &
            array_shape = [ f_out_per_head * 2 * this%num_heads, 1 ] &
       )
       call this%params(2*t)%set_requires_grad(.true.)
       this%params(2*t)%is_sample_dependent = .false.
       this%params(2*t)%is_temporary = .false.
       this%params(2*t)%fix_pointer = .true.
    end do

    ! Initialise weight matrices and attention parameters
    do t = 1, this%num_time_steps
       if(this%concat_heads) then
          f_out_per_head = this%num_vertex_features(t) / this%num_heads
       else
          f_out_per_head = this%num_vertex_features(t)
       end if
       call this%kernel_init%initialise( &
            this%params(2*t-1)%val(:,1), &
            fan_in = this%num_vertex_features(t-1), &
            fan_out = f_out_per_head * this%num_heads, &
            spacing = [ f_out_per_head * this%num_heads ] &
       )
       ! Initialise attention: slice the flat vector into per-head segments
       do k = 1, 2 * this%num_heads
          call this%kernel_init%initialise( &
               this%params(2*t)%val( &
                    (k-1)*f_out_per_head+1:k*f_out_per_head, 1), &
               fan_in = f_out_per_head, &
               fan_out = 1, &
               spacing = [ f_out_per_head ] &
          )
       end do
    end do

    if(allocated(this%output)) deallocate(this%output)

  end subroutine init_gat
!###############################################################################


!###############################################################################
  subroutine print_to_unit_gat(this, unit)
    use coreutils, only: to_upper
    implicit none

    class(gat_msgpass_layer_type), intent(in) :: this
    integer, intent(in) :: unit

    integer :: t
    character(100) :: fmt

    write(unit,'(3X,"NUM_TIME_STEPS = ",I0)') this%num_time_steps
    write(unit,'(3X,"NUM_HEADS = ",I0)') this%num_heads
    write(unit,'(3X,"CONCAT_HEADS = ",L1)') this%concat_heads
    write(unit,'(3X,"NEGATIVE_SLOPE = ",E16.8E2)') this%negative_slope
    write(fmt,'("(3X,""NUM_VERTEX_FEATURES ="",",I0,"(1X,I0))")') &
         this%num_time_steps + 1
    write(unit,fmt) this%num_vertex_features

    if(this%activation%name .ne. 'none')then
       call this%activation%print_to_unit(unit)
    end if

    write(unit,'("WEIGHTS")')
    do t = 1, this%num_time_steps, 1
       write(unit,'(5(E16.8E2))') this%params(2*t-1)%val
    end do
    write(unit,'("END WEIGHTS")')

    write(unit,'("ATTN_WEIGHTS")')
    do t = 1, this%num_time_steps, 1
       write(unit,'(5(E16.8E2))') this%params(2*t)%val
    end do
    write(unit,'("END ATTN_WEIGHTS")')

  end subroutine print_to_unit_gat
!###############################################################################


!###############################################################################
  subroutine read_gat(this, unit, verbose)
    use athena__tools_infile, only: assign_val, assign_vec, get_val, move
    use coreutils, only: to_lower, to_upper, icount
    use athena__activation, only: read_activation
    use athena__initialiser, only: initialiser_setup
    implicit none

    class(gat_msgpass_layer_type), intent(inout) :: this
    integer, intent(in) :: unit
    integer, optional, intent(in) :: verbose

    integer :: stat
    integer :: verbose_ = 0
    integer :: t, j, k, c, itmp1, iline
    integer :: num_time_steps = 0
    integer :: num_heads = 1
    logical :: concat_heads = .true.
    real(real32) :: negative_slope = 0.2_real32
    character(14) :: kernel_initialiser_name=''
    character(20) :: activation_name=''
    class(base_actv_type), allocatable :: activation
    class(base_init_type), allocatable :: kernel_initialiser
    integer, dimension(:), allocatable :: num_vertex_features
    character(256) :: buffer, tag, err_msg
    real(real32), allocatable, dimension(:) :: data_list
    integer :: param_line, final_line, attn_param_line
    logical :: ltmp

    if(present(verbose)) verbose_ = verbose

    iline = 0
    param_line = 0
    attn_param_line = 0
    final_line = 0
    tag_loop: do

       read(unit,'(A)',iostat=stat) buffer
       if(stat.ne.0)then
          write(err_msg,'("file encountered error (EoF?) before END ",A)') &
               to_upper(this%name)
          call stop_program(err_msg)
          return
       end if
       if(trim(adjustl(buffer)).eq."") cycle tag_loop

       if(trim(adjustl(buffer)).eq."END "//to_upper(trim(this%name)))then
          final_line = iline
          backspace(unit)
          exit tag_loop
       end if
       iline = iline + 1

       tag=trim(adjustl(buffer))
       if(scan(buffer,"=").ne.0) tag=trim(tag(:scan(tag,"=")-1))

       select case(trim(tag))
       case("NUM_TIME_STEPS")
          call assign_val(buffer, num_time_steps, itmp1)
       case("NUM_HEADS")
          call assign_val(buffer, num_heads, itmp1)
       case("CONCAT_HEADS")
          call assign_val(buffer, ltmp, itmp1)
          concat_heads = ltmp
       case("NEGATIVE_SLOPE")
          call assign_val(buffer, negative_slope, itmp1)
       case("NUM_VERTEX_FEATURES")
          itmp1 = icount(get_val(buffer))
          allocate(num_vertex_features(itmp1), source=0)
          call assign_vec(buffer, num_vertex_features, itmp1)
       case("ACTIVATION")
          iline = iline - 1
          backspace(unit)
          activation = read_activation(unit, iline)
       case("KERNEL_INITIALISER", "KERNEL_INIT", "KERNEL_INITIALisER")
          call assign_val(buffer, kernel_initialiser_name, itmp1)
       case("WEIGHTS")
          kernel_initialiser_name = 'zeros'
          param_line = iline
       case("ATTN_WEIGHTS")
          attn_param_line = iline
       case default
          if(scan(to_lower(trim(adjustl(buffer))),&
               'abcdfghijklmnopqrstuvwxyz').eq.0)then
             cycle tag_loop
          elseif(tag(:3).eq.'END')then
             cycle tag_loop
          end if
          write(err_msg,'("Unrecognised line in input file: ",A)') &
               trim(adjustl(buffer))
          call stop_program(err_msg)
          return
       end select
    end do tag_loop
    kernel_initialiser = initialiser_setup(kernel_initialiser_name)

    if(num_time_steps.gt.0 .and. &
         num_time_steps.ne.size(num_vertex_features,1)-1)then
       write(err_msg,'("NUM_TIME_STEPS = ",I0," does not match length of "// &
            &"NUM_VERTEX_FEATURES = ",I0)') num_time_steps, &
            size(num_vertex_features,1)-1
       call stop_program(err_msg)
       return
    end if

    this%num_heads = num_heads
    this%concat_heads = concat_heads
    this%negative_slope = negative_slope
    call this%set_hyperparams( &
         num_time_steps = num_time_steps, &
         num_vertex_features = num_vertex_features, &
         activation = activation, &
         kernel_initialiser = kernel_initialiser, &
         verbose = verbose_ &
    )
    call this%init(input_shape=[this%num_vertex_features(0), 0])

    ! Read weight parameters
    if(param_line.eq.0)then
       write(0,*) "WARNING: WEIGHTS card in "// &
            to_upper(trim(this%name))//" not found"
    else
       call move(unit, param_line - iline, iostat=stat)
       do t = 1, this%num_time_steps
          allocate(data_list(size(this%params(2*t-1)%val)), source=0._real32)
          c = 1
          k = 1
          do while(c.le.size(data_list))
             read(unit,'(A)',iostat=stat) buffer
             if(stat.ne.0) exit
             k = icount(buffer)
             read(buffer,*,iostat=stat) (data_list(j),j=c,c+k-1)
             c = c + k
          end do
          this%params(2*t-1)%val(:,1) = data_list
          deallocate(data_list)
       end do
       read(unit,'(A)') buffer
       if(trim(adjustl(buffer)).ne."END WEIGHTS")then
          call stop_program("END WEIGHTS not where expected")
          return
       end if
    end if

    ! Read attention weight parameters
    if(attn_param_line.ne.0)then
       do t = 1, this%num_time_steps
          allocate(data_list(size(this%params(2*t)%val)), source=0._real32)
          c = 1
          k = 1
          do while(c.le.size(data_list))
             read(unit,'(A)',iostat=stat) buffer
             if(stat.ne.0) exit
             k = icount(buffer)
             read(buffer,*,iostat=stat) (data_list(j),j=c,c+k-1)
             c = c + k
          end do
          this%params(2*t)%val(:,1) = data_list
          deallocate(data_list)
       end do
       read(unit,'(A)') buffer
       if(trim(adjustl(buffer)).ne."END ATTN_WEIGHTS")then
          call stop_program("END ATTN_WEIGHTS not where expected")
          return
       end if
    end if

    read(unit,'(A)') buffer
    if(trim(adjustl(buffer)).ne."END "//to_upper(trim(this%name)))then
       write(err_msg,'("END ",A," not where expected")') to_upper(this%name)
       call stop_program(err_msg)
       return
    end if

  end subroutine read_gat
!###############################################################################


!###############################################################################
  function read_gat_msgpass_layer(unit, verbose) result(layer)
    !! Read GAT message passing layer from file and return layer
    implicit none
    integer, intent(in) :: unit
    integer, optional, intent(in) :: verbose
    class(base_layer_type), allocatable :: layer

    integer :: verbose_ = 0

    if(present(verbose)) verbose_ = verbose
    allocate(layer, source = gat_msgpass_layer_type( &
         num_time_steps = 1, &
         num_vertex_features = [ 0, 0 ] &
    ))
    call layer%read(unit, verbose=verbose_)

  end function read_gat_msgpass_layer
!###############################################################################


!###############################################################################
  subroutine update_message_gat(this, input)
    !! Update the message using multi-head attention
    !!
    !! For each sample and time step:
    !! 1. Project features: Wh = W * h (for all heads simultaneously)
    !! 2. Call gat_propagate which handles multi-head attention internally
    !! 3. Apply activation function
    implicit none

    class(gat_msgpass_layer_type), intent(inout), target :: this
    class(array_type), dimension(:,:), intent(in), target :: input

    integer :: s, t, f_out_per_head
    type(array_type), pointer :: ptr_in, ptr_proj, ptr_gat, ptr_activated

    if(allocated(this%output))then
       if(size(this%output,2).ne.size(input,2))then
          deallocate(this%output)
          allocate(this%output(1,size(input,2)))
       end if
    else
       allocate(this%output(1,size(input,2)))
    end if

    do s = 1, size(input,2)
       ptr_in => input(1,s)
       do t = 1, this%num_time_steps
          if(this%concat_heads) then
             f_out_per_head = this%num_vertex_features(t) / this%num_heads
          else
             f_out_per_head = this%num_vertex_features(t)
          end if

          ! Project all features: Wh = W * h
          ! params(2*t-1) shape: [f_out_per_head * num_heads, F_in, 1]
          ! ptr_in shape: [F_in, N]
          ! result shape: [f_out_per_head * num_heads, N]
          ptr_proj => matmul( this%params(2*t-1), ptr_in )

          ! Multi-head attention-weighted propagation
          ! params(2*t) is the attention array_type with
          ! val(f_out_per_head * 2 * num_heads, 1)
          ptr_gat => gat_propagate( &
               ptr_proj, &
               this%params(2*t), &
               this%graph(s)%adj_ia, this%graph(s)%adj_ja, &
               this%negative_slope, &
               this%num_heads, this%concat_heads &
          )

          ! Apply activation
          ptr_activated => this%activation%apply( ptr_gat )
          ptr_in => ptr_activated
       end do
       call this%output(1,s)%zero_grad()
       call this%output(1,s)%assign_and_deallocate_source(ptr_in)
       this%output(1,s)%is_temporary = .false.
    end do

  end subroutine update_message_gat
!###############################################################################


!###############################################################################
  subroutine update_readout_gat(this)
    !! Update the readout (empty for node-level output)
    implicit none
    ! Arguments
    class(gat_msgpass_layer_type), intent(inout), target :: this
    ! Node-level output, no graph-level readout needed
  end subroutine update_readout_gat
!###############################################################################

end module athena__gat_msgpass_layer
