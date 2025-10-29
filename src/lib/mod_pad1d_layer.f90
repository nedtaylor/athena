module athena__pad1d_layer
  !! Module containing implementation of a 1D padding layer
  use coreutils, only: real32, stop_program
  use athena__base_layer, only: pad_layer_type, base_layer_type
  use diffstruc, only: array_type
  use athena__diffstruc_extd, only: pad1d
  implicit none


  private

  public :: pad1d_layer_type
  public :: read_pad1d_layer


  type, extends(pad_layer_type) :: pad1d_layer_type
     !! Type for 1D padding layer with overloaded procedures
   contains
     procedure, pass(this) :: set_hyperparams => set_hyperparams_pad1d
     !! Set hyperparameters for 1D padding layer
     procedure, pass(this) :: set_batch_size => set_batch_size_pad1d
     !! Set batch size for 1D padding layer
     procedure, pass(this) :: read => read_pad1d
     !! Read 1D padding layer from file

     procedure, pass(this) :: forward_derived => forward_derived_pad1d
     !! Forward propagation derived type handler

  end type pad1d_layer_type

  interface pad1d_layer_type
     !! Interface for setting up the 1D padding layer
     module function layer_setup( &
          padding, method, &
          input_shape, batch_size, &
          verbose &
     ) result(layer)
       !! Set up the 1D padding layer
       integer, dimension(:), intent(in) :: padding
       !! Padding sizes
       character(*), intent(in) :: method
       !! Padding method
       integer, dimension(:), optional, intent(in) :: input_shape
       !! Input shape
       integer, optional, intent(in) :: batch_size
       !! Batch size
       integer, optional, intent(in) :: verbose
       !! Verbosity level
       type(pad1d_layer_type) :: layer
       !! Instance of the 1D padding layer
     end function layer_setup
  end interface pad1d_layer_type



contains

!###############################################################################
  module function layer_setup( &
       padding, method, &
       input_shape, batch_size, &
       verbose &
  ) result(layer)
    !! Set up the 1D padding layer
    implicit none

    ! Arguments
    integer, dimension(:), intent(in) :: padding
    !! Padding sizes
    character(*), intent(in) :: method
    !! Padding method
    integer, dimension(:), optional, intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: batch_size
    !! Batch size
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    type(pad1d_layer_type) :: layer
    !! Instance of the 1D padding layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level
    integer, dimension(1) :: padding_1d
    !! 1D padding sizes

    if(present(verbose)) verbose_ = verbose

    !---------------------------------------------------------------------------
    ! Initialise padding sizes
    !---------------------------------------------------------------------------
    select case(size(padding))
    case(1)
       padding_1d = padding
    case default
       write(*,*) size(padding)
       write(*,*) padding
       call stop_program("Invalid padding size")
    end select


    !---------------------------------------------------------------------------
    ! Set hyperparameters
    !---------------------------------------------------------------------------
    call layer%set_hyperparams(padding=padding_1d, method=method, verbose=verbose_)


    !---------------------------------------------------------------------------
    ! Initialise batch size
    !---------------------------------------------------------------------------
    if(present(batch_size)) layer%batch_size = batch_size


    !---------------------------------------------------------------------------
    ! Initialise layer shape
    !---------------------------------------------------------------------------
    if(present(input_shape)) call layer%init(input_shape=input_shape)

  end function layer_setup
!###############################################################################


!###############################################################################
  subroutine set_hyperparams_pad1d(this, padding, method, verbose)
    !! Set hyperparameters for 1D padding layer
    use athena__misc, only: to_lower
    implicit none

    ! Arguments
    class(pad1d_layer_type), intent(inout) :: this
    !! Instance of the 1D padding layer
    integer, dimension(1), intent(in) :: padding
    !! Padding sizes
    character(*), intent(in) :: method
    !! Padding method
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    this%name = "pad1d"
    this%type = "pad"
    this%input_rank = 2
    this%output_rank = 2
    this%pad = padding
    if(allocated(this%facets)) deallocate(this%facets)
    allocate(this%facets(this%input_rank - 1))
    this%facets(1)%rank = 1
    this%facets(1)%nfixed_dims = 1
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

  end subroutine set_hyperparams_pad1d
!###############################################################################


!###############################################################################
  subroutine set_batch_size_pad1d(this, batch_size, verbose)
    !! Set batch size for 1D padding layer
    implicit none

    ! Arguments
    class(pad1d_layer_type), intent(inout), target :: this
    !! Instance of the 1D padding layer
    integer, intent(in) :: batch_size
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
    this%batch_size = batch_size


    !---------------------------------------------------------------------------
    ! Allocate arrays
    !---------------------------------------------------------------------------
    if(allocated(this%input_shape))then
       if(this%use_graph_input)then
          call stop_program("Graph input not supported for 1D padding layer")
          return
       end if
       if(allocated(this%output)) deallocate(this%output)
       allocate( this%output(1,1) )
       call this%output(1,1)%allocate( &
            array_shape = [ &
                 this%output_shape(1), this%num_channels, &
                 this%batch_size ], &
            source=0._real32 &
       )
    end if

  end subroutine set_batch_size_pad1d
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine read_pad1d(this, unit, verbose)
    !! Read 1D padding layer from file
    use athena__tools_infile, only: assign_val, assign_vec
    use athena__misc, only: to_lower, to_upper, icount
    implicit none

    ! Arguments
    class(pad1d_layer_type), intent(inout) :: this
    !! Instance of the 1D padding layer
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
    integer, dimension(1) :: padding
    !! Padding sizes
    integer, dimension(2) :: input_shape
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

  end subroutine read_pad1d
!###############################################################################


!###############################################################################
  function read_pad1d_layer(unit, verbose) result(layer)
    !! Read 1D padding layer from file and return layer
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! File unit
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Instance of the 1D padding layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level

    if(present(verbose)) verbose_ = verbose
    allocate(layer, source=pad1d_layer_type(padding=[0], method="none"))
    call layer%read(unit, verbose=verbose_)

  end function read_pad1d_layer
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine fill_edge_region(this, input, output, orig, dest)
    !! Fill an edge region based on padding method
    implicit none

    ! Arguments
    class(pad1d_layer_type), intent(in) :: this
    real(real32), dimension(:,:,:), intent(in) :: input
    real(real32), dimension(:,:,:), intent(inout) :: output
    integer, dimension(2,1), intent(in) :: orig
    integer, dimension(2,1), intent(in) :: dest

    ! Local variables
    integer :: step

    select case(this%imethod)
    case(3, 4) ! circular or reflection
       step = merge(1, -1, this%imethod .eq. 3)
       output(dest(1,1):dest(2,1), :, :) = input(orig(1,1):orig(2,1):step, :, :)
    case(5) ! replication
       output(dest(1,1):dest(2,1), :, :) = &
            spread(input(orig(1,1), :, :), dim=1, ncopies=this%pad(1))
    end select

  end subroutine fill_edge_region
!###############################################################################


!###############################################################################
  subroutine accumulate_edge_gradient(this, gradient, di_ptr, orig, dest, s, m)
    !! Accumulate gradient from an edge region based on padding method
    implicit none

    ! Arguments
    class(pad1d_layer_type), intent(in) :: this
    real(real32), dimension(:,:,:), intent(in) :: gradient
    real(real32), dimension(:,:,:), intent(inout) :: di_ptr
    integer, dimension(2,1), intent(in) :: orig
    integer, dimension(2,1), intent(in) :: dest
    integer, intent(in) :: s, m

    ! Local variables
    integer :: step

    select case(this%imethod)
    case(3, 4) ! circular or reflection
       step = merge(1, -1, this%imethod .eq. 3)
       di_ptr(orig(1,1):orig(2,1):step, m, s) = &
            di_ptr(orig(1,1):orig(2,1):step, m, s) + &
            gradient(dest(1,1):dest(2,1), m, s)
    case(5) ! replication
       di_ptr(orig(1,1), m, s) = &
            di_ptr(orig(1,1), m, s) + &
            sum( gradient( dest(1,1):dest(2,1), m, s ), dim=1 )
    end select

  end subroutine accumulate_edge_gradient
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine forward_derived_pad1d(this, input)
    !! Forward propagation
    implicit none

    ! Arguments
    class(pad1d_layer_type), intent(inout) :: this
    !! Instance of the 1D padding layer
    class(array_type), dimension(:,:), intent(in) :: input
    !! Input values

    type(array_type), pointer :: ptr


    call this%output(1,1)%zero_grad()
    ptr => pad1d(input(1,1), this%facets(1), this%pad(1), this%imethod)
    call this%output(1,1)%assign_and_deallocate_source(ptr)

  end subroutine forward_derived_pad1d
!###############################################################################

end module athena__pad1d_layer
