module athena__pad3d_layer
  !! Module containing implementation of a 3D padding layer
  !!
  !! This module implements padding for 3D volumetric data, adding values
  !! around boundaries in all three spatial dimensions.
  !!
  !! Operation: Extends volumetric dimensions at boundaries
  !!   Adds padding in width, height, and depth dimensions
  !!
  !! Padding modes:
  !!   - 'constant': pad with fixed value (typically 0)
  !!   - 'replicate': repeat edge values
  !!   - 'reflect': mirror values at boundaries
  !!
  !! Common use: Preserve spatial dimensions in 3D convolutions,
  !! handle boundary effects in video/medical imaging CNNs
  !! Shape: (W,H,D,C) -> (W+p_w, H+p_h, D+p_d, C)
  use coreutils, only: real32, stop_program
  use athena__base_layer, only: pad_layer_type, base_layer_type
  use diffstruc, only: array_type
  use athena__diffstruc_extd, only: pad3d
  implicit none


  private

  public :: pad3d_layer_type
  public :: read_pad3d_layer


  type, extends(pad_layer_type) :: pad3d_layer_type
     !! Type for 3D padding layer with overloaded procedures
   contains
     procedure, pass(this) :: set_hyperparams => set_hyperparams_pad3d
     !! Set hyperparameters for 3D padding layer
     procedure, pass(this) :: read => read_pad3d
     !! Read 3D padding layer from file

     procedure, pass(this) :: forward => forward_pad3d
     !! Forward propagation derived type handler

  end type pad3d_layer_type

  interface pad3d_layer_type
     !! Interface for setting up the 3D padding layer
     module function layer_setup( &
          padding, method, &
          input_shape, &
          verbose &
     ) result(layer)
       !! Set up the 3D padding layer
       integer, dimension(:), intent(in) :: padding
       !! Padding sizes
       character(*), intent(in) :: method
       !! Padding method
       integer, dimension(:), optional, intent(in) :: input_shape
       !! Input shape
       integer, optional, intent(in) :: verbose
       !! Verbosity level
       type(pad3d_layer_type) :: layer
       !! Instance of the 3D padding layer
     end function layer_setup
  end interface pad3d_layer_type



contains

!###############################################################################
  module function layer_setup( &
       padding, method, &
       input_shape, &
       verbose &
  ) result(layer)
    !! Set up the 3D padding layer
    implicit none

    ! Arguments
    integer, dimension(:), intent(in) :: padding
    !! Padding sizes
    character(*), intent(in) :: method
    !! Padding method
    integer, dimension(:), optional, intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    type(pad3d_layer_type) :: layer
    !! Instance of the 3D padding layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level
    integer, dimension(3) :: padding_3d
    !! 3D padding sizes

    if(present(verbose)) verbose_ = verbose

    !---------------------------------------------------------------------------
    ! Initialise padding sizes
    !---------------------------------------------------------------------------
    select case(size(padding))
    case(1)
       padding_3d = [padding(1), padding(1), padding(1)]
    case(3)
       padding_3d = padding
    case default
       call stop_program("Invalid padding size")
    end select


    !---------------------------------------------------------------------------
    ! Set hyperparameters
    !---------------------------------------------------------------------------
    call layer%set_hyperparams(padding=padding_3d, method=method, verbose=verbose_)


    !---------------------------------------------------------------------------
    ! Initialise layer shape
    !---------------------------------------------------------------------------
    if(present(input_shape)) call layer%init(input_shape=input_shape)

  end function layer_setup
!###############################################################################


!###############################################################################
  subroutine set_hyperparams_pad3d(this, padding, method, verbose)
    !! Set hyperparameters for 3D padding layer
    use coreutils, only: to_lower
    implicit none

    ! Arguments
    class(pad3d_layer_type), intent(inout) :: this
    !! Instance of the 3D padding layer
    integer, dimension(3), intent(in) :: padding
    !! Padding sizes
    character(*), intent(in) :: method
    !! Padding method
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    this%name = "pad3d"
    this%type = "pad"
    this%input_rank = 4
    this%output_rank = 4
    this%pad = padding
    if(allocated(this%facets)) deallocate(this%facets)
    allocate(this%facets(this%input_rank - 1))
    this%facets(1)%rank = 3
    this%facets(1)%nfixed_dims = 1
    this%facets(2)%rank = 3
    this%facets(2)%nfixed_dims = 2
    this%facets(3)%rank = 3
    this%facets(3)%nfixed_dims = 3
    select case(trim(adjustl(to_lower(method))))
    case("valid", "none")
       this%imethod = 0
    case("same", "zero", "constant", "const")
       this%imethod = 1
    case("full")
       this%imethod = 2
    case("circular", "circ")
       this%imethod = 3
    case("reflection", "reflect", "refl")
       this%imethod = 4
    case("replication", "replicate", "copy", "repl")
       this%imethod = 5
    case default
       call stop_program("Unrecognised padding method :"//method)
       return
    end select
    this%method = trim(adjustl(to_lower(method)))

  end subroutine set_hyperparams_pad3d
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine read_pad3d(this, unit, verbose)
    !! Read 3D padding layer from file
    use athena__tools_infile, only: assign_val, assign_vec
    use coreutils, only: to_lower, to_upper, icount
    implicit none

    ! Arguments
    class(pad3d_layer_type), intent(inout) :: this
    !! Instance of the 3D padding layer
    integer, intent(in) :: unit
    !! File unit
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level
    integer :: stat
    !! File status
    integer :: itmp1
    !! Temporary integer
    integer, dimension(3) :: padding
    !! Padding sizes
    integer, dimension(4) :: input_shape
    !! Input shape
    character(20) :: method
    !! Padding method
    character(256) :: buffer, tag, err_msg
    !! Buffer for reading lines, tag for identifying lines, error message


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
       case("PADDING")
          call assign_vec(buffer, padding, itmp1)
       case("METHOD")
          call assign_val(buffer, method, itmp1)
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
    call this%set_hyperparams(padding=padding, method=method, verbose=verbose_)
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

  end subroutine read_pad3d
!###############################################################################


!###############################################################################
  function read_pad3d_layer(unit, verbose) result(layer)
    !! Read 3D padding layer from file and return layer
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! File unit
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Instance of the 3D padding layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level

    if(present(verbose)) verbose_ = verbose
    allocate(layer, source=pad3d_layer_type(padding=[0,0,0], method="none"))
    call layer%read(unit, verbose=verbose_)

  end function read_pad3d_layer
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine forward_pad3d(this, input)
    !! Forward propagation
    implicit none

    ! Arguments
    class(pad3d_layer_type), intent(inout) :: this
    !! Instance of the 3D padding layer
    class(array_type), dimension(:,:), intent(in) :: input
    !! Input values

    ! Local variables
    type(array_type), pointer :: ptr
    !! Pointer array


    call this%output(1,1)%zero_grad()
    ptr => pad3d(input(1,1), this%facets, this%pad, this%imethod)
    call this%output(1,1)%assign_and_deallocate_source(ptr)
    this%output(1,1)%is_temporary = .false.

  end subroutine forward_pad3d
!###############################################################################

end module athena__pad3d_layer
