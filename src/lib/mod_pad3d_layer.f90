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
       allocate( this%output(1,1), source = array5d_type() )
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
    integer :: i, j
    !! Loop indices
    integer :: idim
    !! Dimension index
    integer, dimension(2) :: bound_store
    !! Temporary storage for bounds
    integer, dimension(2,3) :: orig_bound, dest_bound
    !! Bounds for input and output arrays


    select type(output => this%output(1,1))
    type is (array5d_type)
       dim_loop: do idim = 1, 3
          dest_bound = this%dest_bound
          orig_bound = this%dest_bound
          dest_bound(:,idim) = [ &
               this%dest_bound(1,idim), &
               this%orig_bound(1,idim) - 1 &
          ]
          ! Assign padding values based on method
          select case(this%imethod)
          case(3) ! circular
             orig_bound(:,idim) = [ &
                  this%orig_bound(2,idim) - this%pad(idim) + 1, &
                  this%orig_bound(2,idim) &
             ]
          case(4) ! reflection
             orig_bound(:,idim) = [ &
                  this%orig_bound(1,idim) + 1, &
                  this%orig_bound(1,idim) + this%pad(idim) &
             ]
          case(5) ! replication
             output%val_ptr(:this%pad(1),:,:,:,:) = spread(input( &
                  this%orig_bound(1,1),:,:,:,: &
             ), dim=1, ncopies=this%pad(1))
             output%val_ptr( &
                  this%output_shape(1) - this%pad(1) + 1 : &
                  this%output_shape(1), :, :, :, : &
             ) = &
                  spread(input( &
                       this%orig_bound(2,1),:,:,:,: &
                  ), dim=1, ncopies=this%pad(1))

             output%val_ptr(:,:this%pad(2),:,:,:) = spread(input( &
                  :,this%orig_bound(1,2),:,:,: &
             ), dim=2, ncopies=this%pad(2))
             output%val_ptr( &
                  :, &
                  this%output_shape(2) - this%pad(2) + 1 : &
                  this%output_shape(2), :, :, : &
             ) = &
                  spread(input( &
                       :,this%orig_bound(2,2),:,:,: &
                  ), dim=2, ncopies=this%pad(2))

             output%val_ptr(:,:,:this%pad(3),:,:) = spread(input( &
                  :,:,this%orig_bound(1,3),:,: &
             ), dim=3, ncopies=this%pad(3))
             output%val_ptr( &
                  :, :, &
                  this%output_shape(3) - this%pad(3) + 1 : &
                  this%output_shape(3), :, : &
             ) = &
                  spread(input( &
                       :,:,this%orig_bound(2,3),:,: &
                  ), dim=3, ncopies=this%pad(3))
             exit dim_loop
          case default
             output%val_ptr(:,:,:,:,:) = 0._real32
             exit dim_loop
          end select

          lr_loop: do j = 1, 2 ! 1 = left padding, 2 = right padding

             output%val_ptr( &
                  dest_bound(1,1):dest_bound(2,1), &
                  dest_bound(1,2):dest_bound(2,2), &
                  dest_bound(1,3):dest_bound(2,3), :, : &
             ) = input( &
                  orig_bound(1,1):orig_bound(2,1), &
                  orig_bound(1,2):orig_bound(2,2), &
                  orig_bound(1,3):orig_bound(2,3), :, : &
             )
             if(j.eq.2) exit lr_loop
             bound_store(:) = dest_bound(:,idim)
             select case(this%imethod)
             case(3) ! circular
                dest_bound(:,idim) = orig_bound(:,idim) + this%pad(i)
                orig_bound(:,idim) = bound_store(:) + this%pad(i)
             case(4) ! reflection
                dest_bound(:,idim) = &
                     orig_bound(:,idim) + this%input_shape(idim) - 1
                orig_bound(:,idim) = bound_store(:) + this%input_shape(idim) - 1
             case(5) ! replication
                dest_bound(:,idim) = orig_bound(:,idim) + this%input_shape(idim)
                orig_bound(:,idim) = bound_store(:) + this%input_shape(idim)
             end select
          end do lr_loop
       end do dim_loop

       output%val_ptr( &
            this%pad(1)+1:this%pad(1)+this%input_shape(1), &
            this%pad(2)+1:this%pad(2)+this%input_shape(2), &
            this%pad(3)+1:this%pad(3)+this%input_shape(3), :, : &
       ) = input
    end select

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
    integer :: i, j, f, s, m, x, y, z
    !! Loop indices
    integer, dimension(3) :: step
    !! Step sizes for reflection
    integer, dimension(2,3) :: orig_bound, dest_bound
    !! Bounds for input and output arrays


    select type(di => this%di(1,1))
    type is (array5d_type)
       di%val_ptr(:,:,:,:,:) = &
            gradient( &
                 this%pad(1)+1:this%pad(1)+this%input_shape(1), &
                 this%pad(2)+1:this%pad(2)+this%input_shape(2), &
                 this%pad(3)+1:this%pad(3)+this%input_shape(3), :, : &
            )

       select case(this%imethod)
       case(3) ! circular
          do i = 1, 3
             orig_bound = this%orig_bound
             dest_bound = this%dest_bound
             do j = 1, 2 ! 1 = left padding, 2 = right padding
                select case(j)
                case(1)
                   orig_bound(:,i) = [ 1, this%pad(i) ]
                   dest_bound(:,i) = [ &
                        this%dest_bound(2,i) - this%pad(i) + 1, &
                        this%dest_bound(2,i) &
                   ]
                case(2)
                   orig_bound(:,i) = [ &
                        this%orig_bound(2,i) - this%pad(i) + 1, &
                        this%orig_bound(2,i) &
                   ]
                   dest_bound(:,i) = [ 1, this%pad(i) ]
                end select

                di%val_ptr( &
                     orig_bound(1,1):orig_bound(2,1), &
                     orig_bound(1,2):orig_bound(2,2), &
                     orig_bound(1,3):orig_bound(2,3), :, : &
                ) = di%val_ptr( &
                     orig_bound(1,1):orig_bound(2,1), &
                     orig_bound(1,2):orig_bound(2,2), &
                     orig_bound(1,3):orig_bound(2,3), :, : &
                ) + gradient( &
                     dest_bound(1,1):dest_bound(2,1), &
                     dest_bound(1,2):dest_bound(2,2), &
                     dest_bound(1,3):dest_bound(2,3), :, : &
                )
             end do
          end do
       case(4) ! reflection
          do i = 1, 3
             orig_bound = this%orig_bound
             dest_bound = this%dest_bound
             step = 1
             step(i) = -1
             do j = 1, 2 ! 1 = left padding, 2 = right padding
                select case(j)
                case(1)
                   orig_bound(:,i) = [ 2, this%pad(i) + 1 ]
                   dest_bound(:,i) = [ this%pad(i), 1 ]
                case(2)
                   orig_bound(:,i) = [ &
                        this%orig_bound(2,i) - this%pad(i), &
                        this%orig_bound(2,i) - 1 &
                   ]
                   dest_bound(:,i) = [ &
                        this%dest_bound(2,i), &
                        this%dest_bound(2,i) - this%pad(i) + 1 &
                   ]
                end select

                di%val_ptr( &
                     orig_bound(1,1):orig_bound(2,1), &
                     orig_bound(1,2):orig_bound(2,2), &
                     orig_bound(1,3):orig_bound(2,3), :, : &
                ) = di%val_ptr( &
                     orig_bound(1,1):orig_bound(2,1), &
                     orig_bound(1,2):orig_bound(2,2), &
                     orig_bound(1,3):orig_bound(2,3), :, : &
                ) + gradient( &
                     dest_bound(1,1):dest_bound(2,1):step(1), &
                     dest_bound(1,2):dest_bound(2,2):step(2), &
                     dest_bound(1,3):dest_bound(2,3):step(3), :, : &
                )
             end do
          end do
       case(5) ! replication

          ! Replicate along corners
          do f = 1, this%facets(3)%num
             do s = 1, this%batch_size
                do m = 1, this%num_channels
                   di%val_ptr( &
                        this%facets(3)%orig_bound(1,f), &
                        this%facets(3)%orig_bound(2,f), &
                        this%facets(3)%orig_bound(3,f), m, s &
                   ) = di%val_ptr( &
                        this%facets(3)%orig_bound(1,f), &
                        this%facets(3)%orig_bound(2,f), &
                        this%facets(3)%orig_bound(3,f), m, s &
                   ) + sum( gradient( &
                             this%facets(3)%dest_bound(1,1,f) : &
                             this%facets(3)%dest_bound(2,1,f), &
                             this%facets(3)%dest_bound(1,2,f) : &
                             this%facets(3)%dest_bound(2,2,f), &
                             this%facets(3)%dest_bound(1,3,f) : &
                             this%facets(3)%dest_bound(2,3,f), m, s &
                        ) )
                end do
             end do
          end do

          ! Replicate along edges
          do f = 1, this%facets(2)%num
             select case(this%facets(2)%dim(f))
             case(1)
                do s = 1, this%batch_size
                   do m = 1, this%num_channels
                      do x = 1, this%input_shape(this%facets(2)%dim(f))
                         di%val_ptr( &
                              x, &
                              this%orig_bound(1,f), &
                              this%orig_bound(2,f), m, s &
                         ) = di%val_ptr( &
                              x, &
                              this%orig_bound(1,f), &
                              this%orig_bound(2,f), m, s &
                         ) + sum( gradient( &
                                   x + this%pad(1), &
                                   this%facets(2)%dest_bound(1,1,f) : &
                                   this%facets(2)%dest_bound(2,1,f), &
                                   this%facets(2)%dest_bound(1,2,f) : &
                                   this%facets(2)%dest_bound(2,2,f), m, s &
                              ) )
                      end do
                   end do
                end do
             case(2)
                do s = 1, this%batch_size
                   do m = 1, this%num_channels
                      do y = 1, this%input_shape(this%facets(2)%dim(f))
                         di%val_ptr( &
                              this%orig_bound(1,f), &
                              y, &
                              this%orig_bound(2,f), m, s &
                         ) = di%val_ptr( &
                              this%orig_bound(1,f), &
                              y, &
                              this%orig_bound(2,f), m, s &
                         ) + sum( gradient( &
                                   this%facets(2)%dest_bound(1,1,f) : &
                                   this%facets(2)%dest_bound(2,1,f), &
                                   y + this%pad(2), &
                                   this%facets(2)%dest_bound(1,2,f) : &
                                   this%facets(2)%dest_bound(2,2,f), m, s &
                              ) )
                      end do
                   end do
                end do
             case(3)
                do s = 1, this%batch_size
                   do m = 1, this%num_channels
                      do z = 1, this%input_shape(this%facets(2)%dim(f))
                         di%val_ptr( &
                              this%orig_bound(1,f), &
                              this%orig_bound(2,f), &
                              z, m, s &
                         ) = di%val_ptr( &
                              this%orig_bound(1,f), &
                              this%orig_bound(2,f), &
                              z, m, s &
                         ) + sum( gradient( &
                                   this%facets(2)%dest_bound(1,1,f) : &
                                   this%facets(2)%dest_bound(2,1,f), &
                                   this%facets(2)%dest_bound(1,2,f) : &
                                   this%facets(2)%dest_bound(2,2,f), &
                                   z + this%pad(3), m, s &
                              ) )
                      end do
                   end do
                end do
             end select
          end do

          ! Replicate along faces
          do f = 1, this%facets(1)%num
             select case(this%facets(1)%dim(f))
             case(1)
                do s = 1, this%batch_size
                   do m = 1, this%num_channels
                      di%val_ptr(this%facets(1)%orig_bound(1,f), :, :, m, s) = &
                           di%val_ptr( &
                                this%facets(1)%orig_bound(1,f), :, :, m, s &
                           ) + &
                           sum( &
                                gradient( &
                                     this%facets(1)%dest_bound(1,1,f) : &
                                     this%facets(1)%dest_bound(2,1,f), &
                                     this%pad(2) : &
                                     this%pad(2) + this%input_shape(2) - 1, &
                                     this%pad(3) : &
                                     this%pad(3) + this%input_shape(3) - 1, &
                                     m, s &
                                ), dim=1 &
                           )
                   end do
                end do
             case(2)
                do s = 1, this%batch_size
                   do m = 1, this%num_channels
                      di%val_ptr(:, this%facets(1)%orig_bound(1,f), :, m, s) = &
                           di%val_ptr( &
                                :, this%facets(1)%orig_bound(1,f), :, m, s &
                           ) + &
                           sum( &
                                gradient( &
                                     this%pad(1) : &
                                     this%pad(1) + this%input_shape(1) - 1, &
                                     this%facets(1)%dest_bound(1,1,f) : &
                                     this%facets(1)%dest_bound(2,1,f), &
                                     this%pad(3) : &
                                     this%pad(3) + this%input_shape(3) - 1, &
                                     m, s &
                                ), dim=2 &
                           )
                   end do
                end do
             case(3)
                do s = 1, this%batch_size
                   do m = 1, this%num_channels
                      di%val_ptr(:, :, this%facets(1)%orig_bound(1,f), m, s) = &
                           di%val_ptr( &
                                :, :, this%facets(1)%orig_bound(1,f), m, s &
                           ) + &
                           sum( &
                                gradient( &
                                     this%pad(1) : &
                                     this%pad(1) + this%input_shape(1) - 1, &
                                     this%pad(2) : &
                                     this%pad(2) + this%input_shape(2) - 1, &
                                     this%facets(1)%dest_bound(1,1,f) : &
                                     this%facets(1)%dest_bound(2,1,f), &
                                     m, s &
                                ), dim=3 &
                           )
                   end do
                end do
             end select
          end do
       end select
    end select

  end subroutine backward_5d
!###############################################################################

end module athena__pad3d_layer
