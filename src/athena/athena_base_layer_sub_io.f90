submodule(athena__base_layer) athena__base_layer_submodule_io
  !! Submodule containing the implementation of the base layer types
  !!
  !! This submodule contains the implementation of the base layer types
  !! used in the ATHENA library. The base layer types are the abstract
  !! types from which all other layer types are derived. The submodule
  !! contains the implementation of the I/O procedures for the base layer
  !!
  use coreutils, only: stop_program, to_upper

contains

!###############################################################################
  module subroutine print_base(this, file, unit, print_header_footer)
    !! Print the layer and wrapping info to a file
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
!###############################################################################


!###############################################################################
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
!###############################################################################


!###############################################################################
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
  module subroutine print_to_unit_batch(this, unit)
    !! Print 3D batch normalisation layer to unit
    implicit none

    ! Arguments
    class(batch_layer_type), intent(in) :: this
    !! Instance of batch normalisation layer
    integer, intent(in) :: unit
    !! File unit

    ! Local variables
    integer :: m
    !! Loop index
    character(100) :: fmt
    !! Format string


    ! Write initial parameters
    !---------------------------------------------------------------------------
    write(fmt,'("(3X,""INPUT_SHAPE = "",",I0,"(1X,I0))")') size(this%input_shape)
    write(unit,fmt) this%input_shape
    write(unit,'(3X,"MOMENTUM = ",F0.9)') this%momentum
    write(unit,'(3X,"EPSILON = ",F0.9)') this%epsilon
    write(unit,'(3X,"NUM_CHANNELS = ",I0)') this%num_channels
    write(unit,'(3X,"GAMMA_INITIALISER = ",A)') trim(this%kernel_init%name)
    write(unit,'(3X,"BETA_INITIALISER = ",A)') trim(this%bias_init%name)
    write(unit,'(3X,"MOVING_MEAN_INITIALISER = ",A)') &
         trim(this%moving_mean_init%name)
    write(unit,'(3X,"MOVING_VARIANCE_INITIALISER = ",A)') &
         trim(this%moving_variance_init%name)
    write(unit,'("GAMMA")')
    do m = 1, this%num_channels
       write(unit,'(5(E16.8E2))') this%params(1)%val(m,1)
    end do
    write(unit,'("END GAMMA")')
    write(unit,'("BETA")')
    do m = 1, this%num_channels
       write(unit,'(5(E16.8E2))') this%params(1)%val(this%num_channels+m,1)
    end do
    write(unit,'("END BETA")')

  end subroutine print_to_unit_batch
!###############################################################################


!###############################################################################
  module subroutine print_to_unit_conv(this, unit)
    !! Print 2D convolutional layer to unit
    implicit none

    ! Arguments
    class(conv_layer_type), intent(in) :: this
    !! Instance of the 2D convolutional layer
    integer, intent(in) :: unit
    !! File unit

    ! Local variables
    integer :: l, i, itmp1, idx
    !! Loop indices
    character(:), allocatable :: padding_type
    !! Padding type
    character(100) :: fmt


    ! Write pad layer if applicable
    !---------------------------------------------------------------------------
    if(allocated(this%pad_layer))then
       call this%pad_layer%print_to_unit(unit)
    end if


    ! Write initial parameters
    !---------------------------------------------------------------------------
    ! write the format string for input shape
    write(fmt,'("(3X,""INPUT_SHAPE ="",",I0,"(1X,I0))")') size(this%input_shape)
    write(unit,fmt) this%input_shape
    write(unit,'(3X,"NUM_FILTERS = ",I0)') this%num_filters
    write(fmt,'("(3X,A,"" ="",",I0,"(1X,I0))")') this%input_rank-1
    if(all(this%knl.eq.this%knl(1)))then
       write(unit,'(3X,"KERNEL_SIZE =",1X,I0)') this%knl(1)
    else
       write(unit,fmt) "KERNEL_SIZE", this%knl
    end if
    if(all(this%stp.eq.this%stp(1)))then
       write(unit,'(3X,"STRIDE =",1X,I0)') this%stp(1)
    else
       write(unit,fmt) "STRIDE", this%stp
    end if
    if(all(this%dil.eq.this%dil(1)))then
       write(unit,'(3X,"DILATION =",1X,I0)') this%dil(1)
    else
       write(unit,fmt) "DILATION", this%dil
    end if

    write(unit,'(3X,"USE_BIAS = ",L1)') this%use_bias
    if(this%activation%name .ne. 'none')then
       call this%activation%print_to_unit(unit)
    end if


    ! Write weights and biases
    !---------------------------------------------------------------------------
    write(unit,'("WEIGHTS")')
    write(unit,'(5(E16.8E2))') this%params(1)%val(:,1)
    write(unit,'(5(E16.8E2))') this%params(2)%val(:,1)
    write(unit,'("END WEIGHTS")')

  end subroutine print_to_unit_conv
!###############################################################################

end submodule athena__base_layer_submodule_io
