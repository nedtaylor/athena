module athena__pad3d_layer
  !! Module containing implementation of a 3D padding layer
  use athena__io_utils, only: stop_program
  use athena__constants, only: real32
  use athena__base_layer, only: pad_layer_type, base_layer_type
  use athena__misc_types, only: array5d_type
  use athena__misc, only: to_lower
  implicit none


  private

  public :: pad3d_layer_type
  public :: read_pad3d_layer


  type, extends(pad_layer_type) :: pad3d_layer_type
     !! Type for 3D padding layer with overloaded procedures
   contains
     procedure, pass(this) :: set_hyperparams => set_hyperparams_pad3d
     !! Set hyperparameters for 3D padding layer
     procedure, pass(this) :: set_batch_size => set_batch_size_pad3d
     !! Set batch size for 3D padding layer
     procedure, pass(this) :: read => read_pad3d
     !! Read 3D padding layer from file
     procedure, pass(this) :: forward  => forward_rank
     !! Forward propagation handler for 3D padding layer
     procedure, pass(this) :: backward => backward_rank
     !! Backward propagation handler for 3D padding layer
     procedure, private, pass(this) :: forward_5d
     !! Forward propagation for 5D input
     procedure, private, pass(this) :: backward_5d
     !! Backward propagation for 5D input
  end type pad3d_layer_type

  interface pad3d_layer_type
     !! Interface for setting up the 3D padding layer
     module function layer_setup( &
          padding, method, &
          input_shape, batch_size, &
          verbose &
     ) result(layer)
       !! Set up the 3D padding layer
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
       type(pad3d_layer_type) :: layer
       !! Instance of the 3D padding layer
     end function layer_setup
  end interface pad3d_layer_type



contains

!###############################################################################
  subroutine forward_rank(this, input)
    !! Forward propagation handler for 3D padding layer
    implicit none

    ! Arguments
    class(pad3d_layer_type), intent(inout) :: this
    !! Instance of the 3D padding layer
    real(real32), dimension(..), intent(in) :: input
    !! Input values

    select rank(input)
    rank(2)
       call forward_5d(this, input)
    rank(5)
       call forward_5d(this, input)
    end select
  end subroutine forward_rank
!###############################################################################


!###############################################################################
  subroutine backward_rank(this, input, gradient)
    !! Backward propagation handler for 3D padding layer
    implicit none

    ! Arguments
    class(pad3d_layer_type), intent(inout) :: this
    !! Instance of the 3D padding layer
    real(real32), dimension(..), intent(in) :: input
    !! Input values
    real(real32), dimension(..), intent(in) :: gradient
    !! Gradient values

    select rank(input)
    rank(2)
       select rank(gradient)
       rank(2)
          call backward_5d(this, input, gradient)
       end select
    rank(5)
       select rank(gradient)
       rank(5)
          call backward_5d(this, input, gradient)
       end select
    end select
  end subroutine backward_rank
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  module function layer_setup( &
       padding, method, &
       input_shape, batch_size, &
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
    integer, optional, intent(in) :: batch_size
    !! Batch size
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
  subroutine set_hyperparams_pad3d(this, padding, method, verbose)
    !! Set hyperparameters for 3D padding layer
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


!###############################################################################
  subroutine set_batch_size_pad3d(this, batch_size, verbose)
    !! Set batch size for 3D padding layer
    implicit none

    ! Arguments
    class(pad3d_layer_type), intent(inout), target :: this
    !! Instance of the 3D padding layer
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
          call stop_program("Graph input not supported for 3D padding layer")
          return
       end if
       if(allocated(this%output)) deallocate(this%output)
       allocate( this%output(1,1) )
       call this%output(1,1)%allocate( &
            array_shape = [ &
                 this%output_shape(1), &
                 this%output_shape(2), &
                 this%output_shape(3), this%num_channels, &
                 this%batch_size ], &
            source=0._real32 &
       )
       if(allocated(this%di)) deallocate(this%di)
       allocate( this%di(1,1), source = array5d_type() )
       call this%di(1,1)%allocate( &
            array_shape = [ &
                 this%input_shape(1), &
                 this%input_shape(2), &
                 this%input_shape(3), &
                 this%input_shape(4), &
                 this%batch_size ], &
            source=0._real32 &
       )
    end if

  end subroutine set_batch_size_pad3d
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine read_pad3d(this, unit, verbose)
    !! Read 3D padding layer from file
    use athena__tools_infile, only: assign_val, assign_vec
    use athena__misc, only: to_lower, to_upper, icount
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
  subroutine fill_corner_region(this, input, output, orig, dest, s, m)
    !! Fill a corner region based on padding method
    implicit none

    ! Arguments
    class(pad3d_layer_type), intent(in) :: this
    real(real32), dimension(:,:,:,:,:), intent(in) :: input
    real(real32), dimension(:,:,:,:,:), intent(inout) :: output
    integer, dimension(2,3), intent(in) :: orig
    integer, dimension(2,3), intent(in) :: dest
    integer, intent(in) :: s, m

    ! Local variables
    integer :: step

    select case(this%imethod)
    case(3, 4) ! circular or reflection
       step = merge(1, -1, this%imethod .eq. 3)
       output(dest(1,1):dest(2,1), dest(1,2):dest(2,2), &
            dest(1,3):dest(2,3), m, s) = &
            input(orig(1,1):orig(2,1):step, orig(1,2):orig(2,2):step, &
                 orig(1,3):orig(2,3):step, m, s)

    case(5) ! replication
       output(dest(1,1):dest(2,1), dest(1,2):dest(2,2), &
            dest(1,3):dest(2,3), m, s) = &
            input(orig(1,1), orig(1,2), orig(1,3), m, s)
    end select

  end subroutine fill_corner_region
!###############################################################################


!###############################################################################
  subroutine fill_edge_region(this, input, output, orig, dest, f, s, m)
    !! Fill an edge region based on padding method
    implicit none

    ! Arguments
    class(pad3d_layer_type), intent(in) :: this
    real(real32), dimension(:,:,:,:,:), intent(in) :: input
    real(real32), dimension(:,:,:,:,:), intent(inout) :: output
    integer, dimension(2,3), intent(in) :: orig
    integer, dimension(2,3), intent(in) :: dest
    integer, intent(in) :: f, s, m

    ! Local variables
    integer :: step1, step2, step3

    select case(this%imethod)
    case(3, 4) ! circular or reflection
       step1 = merge(-1, 1, this%imethod .eq. 4 .and. &
            this%facets(1)%dim(f) .eq. 1)
       step2 = merge(-1, 1, this%imethod .eq. 4 .and. &
            this%facets(1)%dim(f) .eq. 2)
       step3 = merge(-1, 1, this%imethod .eq. 4 .and. &
            this%facets(1)%dim(f) .eq. 3)
       output(dest(1,1):dest(2,1), dest(1,2):dest(2,2), &
            dest(1,3):dest(2,3), m, s) = &
            input(orig(1,1):orig(2,1):step1, orig(1,2):orig(2,2):step2, &
                 orig(1,3):orig(2,3):step3, m, s)
    case(5) ! replication
       select case(this%facets(2)%dim(f))
       case(1) ! Edge along dimension 1
          output(dest(1,1):dest(2,1), dest(1,2):dest(2,2), &
               dest(1,3):dest(2,3), m, s) = &
               spread(spread(input(orig(1,1):orig(2,1), orig(1,2), orig(1,3), m, s), &
                    dim=2, ncopies=this%pad(2)), &
          dim=3, ncopies=this%pad(3))
       case(2) ! Edge along dimension 2
          output(dest(1,1):dest(2,1), dest(1,2):dest(2,2), &
               dest(1,3):dest(2,3), m, s) = &
               spread(spread(input(orig(1,1), orig(1,2):orig(2,2), orig(1,3), m, s), &
                    dim=1, ncopies=this%pad(1)), &
          dim=3, ncopies=this%pad(3))
       case(3) ! Edge along dimension 3
          output(dest(1,1):dest(2,1), dest(1,2):dest(2,2), &
               dest(1,3):dest(2,3), m, s) = &
               spread(spread(input(orig(1,1), orig(1,2), orig(1,3):orig(2,3), m, s), &
                    dim=1, ncopies=this%pad(1)), &
          dim=2, ncopies=this%pad(2))
       end select
    end select

  end subroutine fill_edge_region
!###############################################################################


!###############################################################################
  subroutine fill_face_region(this, input, output, orig, dest, f, s, m)
    !! Fill a face region based on padding method
    implicit none

    ! Arguments
    class(pad3d_layer_type), intent(in) :: this
    real(real32), dimension(:,:,:,:,:), intent(in) :: input
    real(real32), dimension(:,:,:,:,:), intent(inout) :: output
    integer, dimension(2,3), intent(in) :: orig
    integer, dimension(2,3), intent(in) :: dest
    integer, intent(in) :: f, s, m

    ! Local variables
    integer :: step1, step2, step3

    select case(this%imethod)
    case(3, 4) ! circular or reflection
       ! Set steps based on padding method and active dimensions
       step1 = merge(-1, 1, this%imethod .eq. 4 .and. &
            any(this%facets(2)%dim(f) .eq. [1]))
       step2 = merge(-1, 1, this%imethod .eq. 4 .and. &
            any(this%facets(2)%dim(f) .eq. [2]))
       step3 = merge(-1, 1, this%imethod .eq. 4 .and. &
            any(this%facets(2)%dim(f) .eq. [3]))
       output(dest(1,1):dest(2,1), dest(1,2):dest(2,2), &
            dest(1,3):dest(2,3), m, s) = &
            input(orig(1,1):orig(2,1):step1, orig(1,2):orig(2,2):step2, &
                 orig(1,3):orig(2,3):step3, m, s)
    case(5) ! replication
       select case(this%facets(1)%dim(f))
       case(1) ! Face perpendicular to dimension 1
          output(dest(1,1):dest(2,1), dest(1,2):dest(2,2), &
               dest(1,3):dest(2,3), m, s) = &
               spread(input(orig(1,1), :, :, m, s), dim=1, &
                    ncopies=this%pad(1))
       case(2) ! Face perpendicular to dimension 2
          output(dest(1,1):dest(2,1), dest(1,2):dest(2,2), &
               dest(1,3):dest(2,3), m, s) = &
               spread(input(:, orig(1,2), :, m, s), dim=2, &
                    ncopies=this%pad(2))
       case(3) ! Face perpendicular to dimension 3
          output(dest(1,1):dest(2,1), dest(1,2):dest(2,2), &
               dest(1,3):dest(2,3), m, s) = &
               spread(input(:, :, orig(1,3), m, s), dim=3, &
                    ncopies=this%pad(3))
       end select
    end select

  end subroutine fill_face_region
!###############################################################################


!###############################################################################
  subroutine accumulate_corner_gradient(this, gradient, di_ptr, orig, dest, &
       s, m)
    !! Accumulate gradient from a corner region based on padding method
    implicit none

    ! Arguments
    class(pad3d_layer_type), intent(in) :: this
    real(real32), dimension(:,:,:,:,:), intent(in) :: gradient
    real(real32), dimension(:,:,:,:,:), intent(inout) :: di_ptr
    integer, dimension(2,3), intent(in) :: orig
    integer, dimension(2,3), intent(in) :: dest
    integer, intent(in) :: s, m

    ! Local variables
    integer :: step

    select case(this%imethod)
    case(3, 4) ! circular or reflection
       step = merge(1, -1, this%imethod .eq. 3)
       di_ptr(orig(1,1):orig(2,1):step, orig(1,2):orig(2,2):step, &
            orig(1,3):orig(2,3):step, m, s) = &
            di_ptr(orig(1,1):orig(2,1):step, orig(1,2):orig(2,2):step, &
                 orig(1,3):orig(2,3):step, m, s) + &
            gradient(dest(1,1):dest(2,1), dest(1,2):dest(2,2), &
                 dest(1,3):dest(2,3), m, s)

    case(5) ! replication
       di_ptr(orig(1,1), orig(1,2), orig(1,3), m, s) = &
            di_ptr(orig(1,1), orig(1,2), orig(1,3), m, s) + &
            sum(gradient(dest(1,1):dest(2,1), dest(1,2):dest(2,2), &
                 dest(1,3):dest(2,3), m, s))
    end select

  end subroutine accumulate_corner_gradient
!###############################################################################


!###############################################################################
  subroutine accumulate_edge_gradient(this, gradient, di_ptr, orig, dest, &
       f, s, m)
    !! Accumulate gradient from an edge region based on padding method
    implicit none

    ! Arguments
    class(pad3d_layer_type), intent(in) :: this
    real(real32), dimension(:,:,:,:,:), intent(in) :: gradient
    real(real32), dimension(:,:,:,:,:), intent(inout) :: di_ptr
    integer, dimension(2,3), intent(in) :: orig
    integer, dimension(2,3), intent(in) :: dest
    integer, intent(in) :: f, s, m

    ! Local variables
    integer :: step1, step2, step3

    select case(this%imethod)
    case(3, 4) ! circular or reflection
       step1 = merge(-1, 1, this%imethod .eq. 4 .and. &
            this%facets(2)%dim(f) .eq. 1)
       step2 = merge(-1, 1, this%imethod .eq. 4 .and. &
            this%facets(2)%dim(f) .eq. 2)
       step3 = merge(-1, 1, this%imethod .eq. 4 .and. &
            this%facets(2)%dim(f) .eq. 3)
       di_ptr(orig(1,1):orig(2,1):step1, orig(1,2):orig(2,2):step2, &
            orig(1,3):orig(2,3):step3, m, s) = &
            di_ptr(orig(1,1):orig(2,1):step1, orig(1,2):orig(2,2):step2, &
                 orig(1,3):orig(2,3):step3, m, s) + &
            gradient(dest(1,1):dest(2,1), dest(1,2):dest(2,2), &
                 dest(1,3):dest(2,3), m, s)
    case(5) ! replication
       select case(this%facets(2)%dim(f))
       case(1) ! Edge along dimension 1
          di_ptr(orig(1,1):orig(2,1), orig(1,2), orig(1,3), m, s) = &
               di_ptr(orig(1,1):orig(2,1), orig(1,2), orig(1,3), m, s) + &
               sum(sum(gradient(dest(1,1):dest(2,1), dest(1,2):dest(2,2), &
                    dest(1,3):dest(2,3), m, s), dim = 3), dim = 2)
       case(2) ! Edge along dimension 2
          di_ptr(orig(1,1), orig(1,2):orig(2,2), orig(1,3), m, s) = &
               di_ptr(orig(1,1), orig(1,2):orig(2,2), orig(1,3), m, s) + &
               sum(sum(gradient(dest(1,1):dest(2,1), dest(1,2):dest(2,2), &
                    dest(1,3):dest(2,3), m, s), dim = 3), dim = 1)
       case(3) ! Edge along dimension 3
          di_ptr(orig(1,1), orig(1,2), orig(1,3):orig(2,3), m, s) = &
               di_ptr(orig(1,1), orig(1,2), orig(1,3):orig(2,3), m, s) + &
               sum(sum(gradient(dest(1,1):dest(2,1), dest(1,2):dest(2,2), &
                    dest(1,3):dest(2,3), m, s), dim = 2), dim = 1)
       end select
    end select

  end subroutine accumulate_edge_gradient
!###############################################################################


!###############################################################################
  subroutine accumulate_face_gradient(this, gradient, di_ptr, orig, dest, &
       f, s, m)
    !! Accumulate gradient from a face region based on padding method
    implicit none

    ! Arguments
    class(pad3d_layer_type), intent(in) :: this
    real(real32), dimension(:,:,:,:,:), intent(in) :: gradient
    real(real32), dimension(:,:,:,:,:), intent(inout) :: di_ptr
    integer, dimension(2,3), intent(in) :: orig
    integer, dimension(2,3), intent(in) :: dest
    integer, intent(in) :: f, s, m

    ! Local variables
    integer :: step1, step2, step3

    select case(this%imethod)
    case(3, 4) ! circular or reflection
       step1 = merge(-1, 1, this%imethod .eq. 4 .and. &
            any(this%facets(1)%dim(f) .eq. [1]))
       step2 = merge(-1, 1, this%imethod .eq. 4 .and. &
            any(this%facets(1)%dim(f) .eq. [2]))
       step3 = merge(-1, 1, this%imethod .eq. 4 .and. &
            any(this%facets(1)%dim(f) .eq. [3]))
       di_ptr(orig(1,1):orig(2,1):step1, orig(1,2):orig(2,2):step2, &
            orig(1,3):orig(2,3):step3, m, s) = &
            di_ptr(orig(1,1):orig(2,1):step1, orig(1,2):orig(2,2):step2, &
                 orig(1,3):orig(2,3):step3, m, s) + &
            gradient(dest(1,1):dest(2,1), dest(1,2):dest(2,2), &
                 dest(1,3):dest(2,3), m, s)
    case(5) ! replication
       select case(this%facets(1)%dim(f))
       case(1) ! Face perpendicular to dimension 1
          di_ptr(orig(1,1), :, :, m, s) = &
               di_ptr(orig(1,1), :, :, m, s) + &
               sum(gradient(dest(1,1):dest(2,1), dest(1,2):dest(2,2), &
                    dest(1,3):dest(2,3), m, s), dim=1)
       case(2) ! Face perpendicular to dimension 2
          di_ptr(:, orig(1,2), :, m, s) = &
               di_ptr(:, orig(1,2), :, m, s) + &
               sum(gradient(dest(1,1):dest(2,1), dest(1,2):dest(2,2), &
                    dest(1,3):dest(2,3), m, s), dim=2)
       case(3) ! Face perpendicular to dimension 3
          di_ptr(:, :, orig(1,3), m, s) = &
               di_ptr(:, :, orig(1,3), m, s) + &
               sum(gradient(dest(1,1):dest(2,1), dest(1,2):dest(2,2), &
                    dest(1,3):dest(2,3), m, s), dim=3)
       end select
    end select

  end subroutine accumulate_face_gradient
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine forward_5d(this, input)
    !! Forward propagation for 5D input
    implicit none

    ! Arguments
    class(pad3d_layer_type), intent(inout) :: this
    !! Instance of the 3D padding layer
    real(real32), &
         dimension( &
              this%input_shape(1), &
              this%input_shape(2), &
              this%input_shape(3), &
              this%num_channels, &
              this%batch_size), &
         intent(in) :: input
    !! Input values

    ! Local variables
    integer :: f, s, m
    !! Loop indices

    !  select type(output => this%output(1,1))
    !  type is (array5d_type)
    !     ! Initialize with zeros for default case
    !     output%val_ptr(:,:,:,:,:) = 0._real32

    !     ! Copy main input region to output
    !     output%val_ptr( &
    !          this%pad(1)+1:this%pad(1)+this%input_shape(1), &
    !          this%pad(2)+1:this%pad(2)+this%input_shape(2), &
    !          this%pad(3)+1:this%pad(3)+this%input_shape(3), :, : &
    !     ) = input

    !     ! Handle padding methods that require boundary filling
    !     if (this%imethod .ge. 3 .and. this%imethod .le. 5) then
    !        ! Process corners (3D corners)
    !        do f = 1, this%facets(3)%num
    !           do s = 1, this%batch_size
    !              do m = 1, this%num_channels
    !                 call fill_corner_region(this, input, output%val_ptr, &
    !                      this%facets(3)%orig_bound(:,:,f), &
    !                      this%facets(3)%dest_bound(:,:,f), s, m)
    !              end do
    !           end do
    !        end do

    !        ! Process faces (2D faces)
    !        do f = 1, this%facets(2)%num
    !           do s = 1, this%batch_size
    !              do m = 1, this%num_channels
    !                 call fill_edge_region(this, input, output%val_ptr, &
    !                      this%facets(2)%orig_bound(:,:,f), &
    !                      this%facets(2)%dest_bound(:,:,f), f, s, m)
    !              end do
    !           end do
    !        end do

    !        ! Process edges (1D edges)
    !        do f = 1, this%facets(1)%num
    !           do s = 1, this%batch_size
    !              do m = 1, this%num_channels
    !                 call fill_face_region(this, input, output%val_ptr, &
    !                      this%facets(1)%orig_bound(:,:,f), &
    !                      this%facets(1)%dest_bound(:,:,f), f, s, m)
    !              end do
    !           end do
    !        end do
    !     end if
    !  end select

  end subroutine forward_5d
!###############################################################################


!###############################################################################
  subroutine backward_5d(this, input, gradient)
    !! Backward propagation for 5D input
    implicit none

    ! Arguments
    class(pad3d_layer_type), intent(inout) :: this
    !! Instance of the 3D padding layer
    real(real32), &
         dimension( &
              this%input_shape(1), &
              this%input_shape(2), &
              this%input_shape(3), &
              this%num_channels, &
              this%batch_size), &
         intent(in) :: input
    !! Input values
    real(real32), &
         dimension(&
              this%output_shape(1), &
              this%output_shape(2), &
              this%output_shape(3), &
              this%num_channels, &
              this%batch_size), &
         intent(in) :: gradient
    !! Gradient values

    ! Local variables
    integer :: f, s, m
    !! Loop indices

    select type(di => this%di(1,1))
    type is (array5d_type)
       ! Copy main gradient region
       di%val_ptr(:,:,:,:,:) = &
            gradient( &
                 this%pad(1)+1:this%pad(1)+this%input_shape(1), &
                 this%pad(2)+1:this%pad(2)+this%input_shape(2), &
                 this%pad(3)+1:this%pad(3)+this%input_shape(3), :, : &
            )

       ! Handle padding methods that require boundary accumulation
       if (this%imethod .ge. 3 .and. this%imethod .le. 5) then
          ! Process corners (3D corners)
          do f = 1, this%facets(3)%num
             do s = 1, this%batch_size
                do m = 1, this%num_channels
                   call accumulate_corner_gradient(this, gradient, di%val_ptr, &
                        this%facets(3)%orig_bound(:,:,f), &
                        this%facets(3)%dest_bound(:,:,f), s, m)
                end do
             end do
          end do

          ! Process faces (2D faces)
          do f = 1, this%facets(2)%num
             do s = 1, this%batch_size
                do m = 1, this%num_channels
                   call accumulate_edge_gradient(this, gradient, di%val_ptr, &
                        this%facets(2)%orig_bound(:,:,f), &
                        this%facets(2)%dest_bound(:,:,f), f, s, m)
                end do
             end do
          end do

          ! Process edges (1D edges)
          do f = 1, this%facets(1)%num
             do s = 1, this%batch_size
                do m = 1, this%num_channels
                   call accumulate_face_gradient(this, gradient, di%val_ptr, &
                        this%facets(1)%orig_bound(:,:,f), &
                        this%facets(1)%dest_bound(:,:,f), f, s, m)
                end do
             end do
          end do
       end if
    end select

  end subroutine backward_5d
!###############################################################################

end module athena__pad3d_layer
