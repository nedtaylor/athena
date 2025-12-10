module athena__dropblock3d_layer
  !! Module containing implementation of a 3D dropblock layer
  !!
  !! This module implements DropBlock regularization for 3D convolutional layers,
  !! dropping contiguous 3D regions (blocks) instead of individual elements.
  !! Extension of 2D DropBlock for volumetric/spatiotemporal data.
  !!
  !! Mathematical operation (training):
  !!   1. Compute drop probability per spatial location:
  !!      gamma = p * (feature_size^3) / (block_size^3 * valid_positions)
  !!   2. Sample Bernoulli mask M_i ~ Bernoulli(gamma)
  !!   3. Expand mask to block_size x block_size x block_size blocks
  !!   4. Apply and normalize:
  !!      y = x * M * (count_elements / count_ones)
  !!
  !! where block_size is the spatial extent of each dropped block in all 3 dims
  !!
  !! Inference: acts as identity (no dropout applied)
  !! \[
  !!   y_i = x_i
  !! \]
  !!
  !! Benefits: Spatial/temporal coherence for 3D CNNs, better for video/volumetric,
  !! removes spatiotemporal semantic information
  !! Typical: block_size=5-7, keep_prob=0.9 for 3D ResNets
  !! Reference: Ghiasi et al. (2018), NeurIPS - https://arxiv.org/abs/1810.12890
  use coreutils, only: real32, stop_program
  use athena__base_layer, only: drop_layer_type, base_layer_type
  use diffstruc, only: array_type, operator(*)
  use athena__misc_types, only: &
       onnx_node_type, onnx_initialiser_type, onnx_tensor_type
  use athena__diffstruc_extd, only: merge_over_channels
  implicit none


  private

  public :: dropblock3d_layer_type
  public :: read_dropblock3d_layer


  type, extends(drop_layer_type) :: dropblock3d_layer_type
     !! Type for 3D dropblock layer with overloaded procedures
     integer :: block_size, half
     !! Block size and half block size
     !! Block size is the width of the block to drop (typical = 5)
     real(real32) :: gamma
     !! Number of activation units to drop
     integer :: num_channels
     !! Number of channels
     logical, allocatable, dimension(:,:,:) :: mask
     !! Mask for dropblock
   contains
     procedure, pass(this) :: set_hyperparams => set_hyperparams_dropblock3d
     !! Set hyperparameters for 3D dropblock layer
     procedure, pass(this) :: init => init_dropblock3d
     !! Initialise 3D dropblock layer
     procedure, pass(this) :: print_to_unit => print_to_unit_dropblock3d
     !! Print 3D dropblock layer to unit
     procedure, pass(this) :: read => read_dropblock3d
     !! Read 3D dropblock layer from file

     procedure, pass(this) :: forward => forward_dropblock3d
     !! Forward propagation derived type handler

     procedure, pass(this) :: generate_mask => generate_bernoulli_mask
     !! Generate Bernoulli mask
  end type dropblock3d_layer_type

  interface dropblock3d_layer_type
     !! Interface for setting up the 3D dropblock layer
     module function layer_setup( &
          rate, block_size, &
          input_shape, &
          verbose ) result(layer)
       !! Set up the 3D dropblock layer
       real(real32), intent(in) :: rate
       !! Drop rate
       integer, intent(in) :: block_size
       !! Block size
       integer, dimension(:), optional, intent(in) :: input_shape
       !! Input shape
       integer, optional, intent(in) :: verbose
       !! Verbosity level
       type(dropblock3d_layer_type) :: layer
       !! Instance of the 3D dropblock layer
     end function layer_setup
  end interface dropblock3d_layer_type



contains

!###############################################################################
  module function layer_setup( &
       rate, block_size, &
       input_shape, &
       verbose ) result(layer)
    !! Set up the 3D dropblock layer
    implicit none

    ! Arguments
    real(real32), intent(in) :: rate
    !! Drop rate
    integer, intent(in) :: block_size
    !! Block size
    integer, dimension(:), optional, intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    type(dropblock3d_layer_type) :: layer
    !! Instance of the 3D dropblock layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level

    if(present(verbose)) verbose_ = verbose

    !---------------------------------------------------------------------------
    ! Initialise hyperparameters
    !---------------------------------------------------------------------------
    call layer%set_hyperparams(rate, block_size, verbose=verbose_)


    !---------------------------------------------------------------------------
    ! Initialise layer shape
    !---------------------------------------------------------------------------
    if(present(input_shape)) call layer%init(input_shape=input_shape)

  end function layer_setup
!###############################################################################


!###############################################################################
  subroutine set_hyperparams_dropblock3d(this, rate, block_size, verbose)
    !! Set hyperparameters for 3D dropblock layer
    implicit none

    ! Arguments
    class(dropblock3d_layer_type), intent(inout) :: this
    !! Instance of the 3D dropblock layer
    real(real32), intent(in) :: rate
    !! Drop rate
    integer, intent(in) :: block_size
    !! Block size
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    this%name = "dropblock3d"
    this%type = "drop"
    this%input_rank = 4
    this%output_rank = 4

    this%rate = rate
    this%block_size = block_size
    this%half = (this%block_size-1)/2

  end subroutine set_hyperparams_dropblock3d
!###############################################################################


!###############################################################################
  subroutine init_dropblock3d(this, input_shape, verbose)
    !! Initialise 3D dropblock layer
    implicit none

    ! Arguments
    class(dropblock3d_layer_type), intent(inout) :: this
    !! Instance of the 3D dropblock layer
    integer, dimension(:), intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level

    !---------------------------------------------------------------------------
    ! initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose


    !---------------------------------------------------------------------------
    ! initialise input shape
    !---------------------------------------------------------------------------
    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)


    !---------------------------------------------------------------------------
    ! set up number of channels, width, height
    !---------------------------------------------------------------------------
    this%num_channels = this%input_shape(4)
    allocate(this%output_shape(2))
    this%output_shape = this%input_shape


    !---------------------------------------------------------------------------
    ! set gamma
    !---------------------------------------------------------------------------
    ! original paper uses keep_prob, we use drop_rate
    ! drop_rate = 1 - keep_prob
    this%gamma = ( this%rate/this%block_size**3._real32 ) * &
         this%input_shape(1) / &
         (this%input_shape(1) - this%block_size + 1._real32) * &
         this%input_shape(2) / &
         (this%input_shape(2) - this%block_size + 1._real32) * &
         this%input_shape(3) / &
         (this%input_shape(3) - this%block_size + 1._real32)
    allocate(this%mask( &
         this%input_shape(1), &
         this%input_shape(2), &
         this%input_shape(3)), source=.true.)


    !---------------------------------------------------------------------------
    ! generate mask
    !---------------------------------------------------------------------------
    call this%generate_mask()


    !---------------------------------------------------------------------------
    ! Allocate arrays
    !---------------------------------------------------------------------------
    if(this%use_graph_input)then
       call stop_program( &
            "Graph input not supported for 3D dropblock layer" &
       )
       return
    end if
    if(allocated(this%output)) deallocate(this%output)
    allocate( this%output(1,1) )

  end subroutine init_dropblock3d
!###############################################################################


!###############################################################################
  subroutine generate_bernoulli_mask(this)
    !! Generate Bernoulli mask
    implicit none

    ! Arguments
    class(dropblock3d_layer_type), intent(inout) :: this
    !! Instance of the 3D dropblock layer

    ! Local variables
    real(real32), allocatable, dimension(:,:,:) :: mask_real
    !! Real mask
    integer :: i, j, k
    !! Loop indices
    integer, dimension(2) :: ilim, jlim, klim
    !! Limits for mask


    ! Generate Bernoulli mask
    !---------------------------------------------------------------------------
    ! assume random number already seeded and don't need to again
    allocate(mask_real(size(this%mask,1), size(this%mask,2), size(this%mask,3)))
    call random_number(mask_real)  ! Generate random values in [0..1]

    this%mask = .true. ! 1 = keep

    !! Apply threshold to create binary mask
    !---------------------------------------------------------------------------
    do k = 1 + this%half, size(this%mask, dim=3) - this%half
       do j = 1 + this%half, size(this%mask, dim=2) - this%half
          do i = 1 + this%half, size(this%mask, dim=1) - this%half
             if(mask_real(i, j, k).lt.this%gamma)then
                ilim(:) = [ &
                     max(i - this%half, lbound(this%mask,1)), &
                     min(i + this%half, ubound(this%mask,1)) ]
                jlim(:) = [ &
                     max(j - this%half, lbound(this%mask,2)), &
                     min(j + this%half, ubound(this%mask,2)) ]
                klim(:) = [ &
                     max(k - this%half, lbound(this%mask,3)), &
                     min(k + this%half, ubound(this%mask,3)) ]
                this%mask( &
                     ilim(1):ilim(2), &
                     jlim(1):jlim(2), &
                     klim(1):klim(2)) = .false. ! 0 = drop
             end if
          end do
       end do
    end do

  end subroutine generate_bernoulli_mask
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine print_to_unit_dropblock3d(this, unit)
    !! Print 3D dropblock layer to unit
    use coreutils, only: to_upper
    implicit none

    ! Arguments
    class(dropblock3d_layer_type), intent(in) :: this
    !! Instance of the 3D dropblock layer
    integer, intent(in) :: unit
    !! File unit


    ! Write initial parameters
    !---------------------------------------------------------------------------
    write(unit,'(3X,"INPUT_SHAPE = ",4(1X,I0))') this%input_shape
    write(unit,'(3X,"RATE = ",F0.9)') this%rate
    write(unit,'(3X,"BLOCK_SIZE = ",I0)') this%block_size

  end subroutine print_to_unit_dropblock3d
!###############################################################################


!###############################################################################
  subroutine read_dropblock3d(this, unit, verbose)
    !! Read 3D dropblock layer from file
    use athena__tools_infile, only: assign_val, assign_vec
    use coreutils, only: to_lower, to_upper, icount
    implicit none

    ! Arguments
    class(dropblock3d_layer_type), intent(inout) :: this
    !! Instance of the 3D dropblock layer
    integer, intent(in) :: unit
    !! File unit
    integer, intent(in), optional :: verbose
    !! Verbosity level

    ! Local variables
    integer :: stat, verbose_ = 0
    !! File status and verbosity level
    integer :: itmp1
    !! Temporary integer
    integer :: block_size
    !! Block size
    real(real32) :: rate
    !! Drop rate
    integer, dimension(4) :: input_shape
    !! Input shape
    character(256) :: buffer, tag, err_msg
    !! Buffer, tag, and error message


    ! Initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose


    ! Loop over tags in layer card
    !---------------------------------------------------------------------------
    tag_loop: do

       ! Check for end of file
       !------------------------------------------------------------------------
       read(unit,'(A)',iostat=stat) buffer
       if(stat.ne.0)then
          write(err_msg,'("file encountered error (EoF?) before END ",A)') &
               to_upper(this%name)
          call stop_program(err_msg)
          return
       end if
       if(trim(adjustl(buffer)).eq."") cycle tag_loop

       ! Check for end of layer card
       !------------------------------------------------------------------------
       if(trim(adjustl(buffer)).eq."END "//to_upper(trim(this%name)))then
          backspace(unit)
          exit tag_loop
       end if

       tag=trim(adjustl(buffer))
       if(scan(buffer,"=").ne.0) tag=trim(tag(:scan(tag,"=")-1))

       ! Read parameters from save file
       !------------------------------------------------------------------------
       select case(trim(tag))
       case("INPUT_SHAPE")
          call assign_vec(buffer, input_shape, itmp1)
       case("RATE")
          call assign_val(buffer, rate, itmp1)
       case("BLOCK_SIZE")
          call assign_val(buffer, block_size, itmp1)
       case default
          ! Don't look for "e" due to scientific notation of numbers
          ! ... i.e. exponent (E+00)
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


    ! Set hyperparameters and initialise layer
    !---------------------------------------------------------------------------
    call this%set_hyperparams( &
         rate = rate, block_size = block_size, &
         verbose = verbose_ &
    )
    call this%init(input_shape = input_shape)


    ! Check for end of layer card
    !---------------------------------------------------------------------------
    read(unit,'(A)') buffer
    if(trim(adjustl(buffer)).ne."END "//to_upper(trim(this%name)))then
       write(0,*) trim(adjustl(buffer))
       write(err_msg,'("END ",A," not where expected")') to_upper(this%name)
       call stop_program(err_msg)
       return
    end if

  end subroutine read_dropblock3d
!###############################################################################


!###############################################################################
  function read_dropblock3d_layer(unit, verbose) result(layer)
    !! Read 3D dropblock layer from file and return layer
    implicit none
    integer, intent(in) :: unit
    !! File unit
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Instance of the 3D dropblock layer

    integer :: verbose_ = 0
    !! Verbosity level

    if(present(verbose)) verbose_ = verbose
    allocate(layer, source=dropblock3d_layer_type(rate=0._real32, block_size=0))
    call layer%read(unit, verbose=verbose_)

  end function read_dropblock3d_layer
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine build_from_onnx_dropblock3d( &
       this, node, initialisers, value_info, verbose &
  )
    !! Read ONNX attributes for 3D dropblock layer
    implicit none

    ! Arguments
    class(dropblock3d_layer_type), intent(inout) :: this
    !! Instance of the 3D dropblock layer
    type(onnx_node_type), intent(in) :: node
    !! ONNX node information
    type(onnx_initialiser_type), dimension(:), intent(in) :: initialisers
    !! ONNX initialiser information
    type(onnx_tensor_type), dimension(:), intent(in) :: value_info
    !! ONNX value info
    integer, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: i
    !! Loop index
    real(real32) :: rate
    !! Dropout rate
    integer :: block_size
    !! Block size
    character(256) :: val
    !! Attribute value

    ! Set default values
    rate = 0.1_real32
    block_size = 7

    ! Parse ONNX attributes
    do i = 1, size(node%attributes)
       val = node%attributes(i)%val
       select case(trim(adjustl(node%attributes(i)%name)))
       case("drop_prob")
          read(val,*) rate
       case("block_size")
          read(val,*) block_size
       case default
          ! Do nothing
          write(0,*) "WARNING: Unrecognised attribute in ONNX &
               &DROPBLOCK3D layer: ", trim(adjustl(node%attributes(i)%name))
       end select
    end do

    ! Check size of initialisers is zero
    if(size(initialisers).ne.0)then
       write(0,*) "WARNING: initialisers not used for ONNX DROPBLOCK3D layer"
    end if

    call this%set_hyperparams( &
         rate = rate, &
         block_size = block_size, &
         verbose = verbose &
    )

  end subroutine build_from_onnx_dropblock3d
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine forward_dropblock3d(this, input)
    !! Forward propagation
    implicit none

    ! Arguments
    class(dropblock3d_layer_type), intent(inout) :: this
    !! Instance of the 3D dropblock layer
    class(array_type), dimension(:,:), intent(in) :: input
    !! Input values

    ! Local variables
    real(real32) :: rtmp1
    !! Temporary variable
    type(array_type), pointer :: ptr
    !! Pointer array


    rtmp1 = 1._real32 - this%rate
    select case(this%inference)
    case(.true.)
       ! Do not perform the drop operation
       ptr => input(1,1) * rtmp1
    case default
       ! Perform the drop operation
       rtmp1 = 1._real32 / rtmp1
       ptr => merge_over_channels( &
            input(1,1), 0._real32, &
            reshape(this%mask, shape = [product(shape(this%mask)), 1]) &
       ) * rtmp1
    end select
    call this%output(1,1)%assign_and_deallocate_source(ptr)
    this%output(1,1)%is_temporary = .false.

  end subroutine forward_dropblock3d
!###############################################################################

end module athena__dropblock3d_layer
