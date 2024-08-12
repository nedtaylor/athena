!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module contains implementation of a 2D average pooling layer
!!!#############################################################################
module avgpool2d_layer
  use constants, only: real32
  use base_layer, only: pool_layer_type
  use custom_types, only: array4d_type
  implicit none
  
  
  type, extends(pool_layer_type) :: avgpool2d_layer_type
   !   real(real32), allocatable, dimension(:,:,:,:) :: output
   !   real(real32), allocatable, dimension(:,:,:,:) :: di ! gradient of input (i.e. delta)
   contains
     procedure, pass(this) :: set_hyperparams => set_hyperparams_avgpool2d
     procedure, pass(this) :: init => init_avgpool2d
     procedure, pass(this) :: set_batch_size => set_batch_size_avgpool2d
     procedure, pass(this) :: print => print_avgpool2d
     procedure, pass(this) :: read => read_avgpool2d
     procedure, pass(this) :: forward  => forward_rank
     procedure, pass(this) :: backward => backward_rank
     procedure, private, pass(this) :: forward_4d
     procedure, private, pass(this) :: backward_4d
  end type avgpool2d_layer_type

  
  interface avgpool2d_layer_type
     module function layer_setup( &
          input_shape, batch_size, &
          pool_size, stride, verbose ) result(layer)
       integer, dimension(:), optional, intent(in) :: input_shape
       integer, optional, intent(in) :: batch_size 
       integer, dimension(..), optional, intent(in) :: pool_size
       integer, dimension(..), optional, intent(in) :: stride
       integer, optional, intent(in) :: verbose
       type(avgpool2d_layer_type) :: layer
     end function layer_setup
  end interface avgpool2d_layer_type


  private
  public :: avgpool2d_layer_type
  public :: read_avgpool2d_layer


contains

!!!#############################################################################
!!! forward propagation assumed rank handler
!!!#############################################################################
  pure subroutine forward_rank(this, input)
    implicit none
    class(avgpool2d_layer_type), intent(inout) :: this
    real(real32), dimension(..), intent(in) :: input

    select rank(input); rank(4)
       call forward_4d(this, input)
    end select
  end subroutine forward_rank
!!!#############################################################################


!!!#############################################################################
!!! backward propagation assumed rank handler
!!!#############################################################################
  pure subroutine backward_rank(this, input, gradient)
    implicit none
    class(avgpool2d_layer_type), intent(inout) :: this
    real(real32), dimension(..), intent(in) :: input
    real(real32), dimension(..), intent(in) :: gradient

    select rank(input); rank(4)
    select rank(gradient); rank(4)
      call backward_4d(this, input, gradient)
    end select
    end select
  end subroutine backward_rank
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! set up layer
!!!#############################################################################
#if defined(GFORTRAN)
  module function layer_setup( &
       input_shape, batch_size, &
       pool_size, stride, verbose) result(layer)
    implicit none
    integer, dimension(:), optional, intent(in) :: input_shape
    integer, optional, intent(in) :: batch_size 
    integer, dimension(..), optional, intent(in) :: pool_size
    integer, dimension(..), optional, intent(in) :: stride
    integer, optional, intent(in) :: verbose
    
    type(avgpool2d_layer_type) :: layer
#else
  module procedure layer_setup
    implicit none
#endif

    integer :: verbose_ = 0
    integer, dimension(2) :: pool_size_, stride_


    if(present(verbose)) verbose_ = verbose

    !!-----------------------------------------------------------------------
    !! set up pool size
    !!-----------------------------------------------------------------------
    if(present(pool_size))then
       select rank(pool_size)
       rank(0)
          pool_size_ = pool_size
       rank(1)
          pool_size_(1) = pool_size(1)
          if(size(pool_size,dim=1).eq.1)then
             pool_size_(2) = pool_size(1)
          elseif(size(pool_size,dim=1).eq.2)then
             pool_size_(2) = pool_size(2)
          end if
       end select
    else
       pool_size_ = 2
    end if


    !!-----------------------------------------------------------------------
    !! set up stride
    !!-----------------------------------------------------------------------
    if(present(stride))then
       select rank(stride)
       rank(0)
          stride_ = stride
       rank(1)
          stride_ = stride(1)
          if(size(stride,dim=1).eq.1)then
             stride_ = stride(1)
          elseif(size(stride,dim=1).eq.2)then
             stride_ = stride(2)
          end if
       end select
    else
       stride_ = 2
    end if


    !!--------------------------------------------------------------------------
    !! set hyperparameters
    !!--------------------------------------------------------------------------
    call layer%set_hyperparams( &
         pool_size=pool_size_, stride=stride_, verbose=verbose_ &
    )


    !!--------------------------------------------------------------------------
    !! initialise batch size
    !!--------------------------------------------------------------------------
    if(present(batch_size)) layer%batch_size = batch_size


    !!--------------------------------------------------------------------------
    !! initialise layer shape
    !!--------------------------------------------------------------------------
    if(present(input_shape)) call layer%init(input_shape=input_shape)

#if defined(GFORTRAN)
  end function layer_setup
#else
  end procedure layer_setup
#endif
!!!#############################################################################


!!!#############################################################################
!!! set hyperparameters
!!!#############################################################################
  subroutine set_hyperparams_avgpool2d(this, pool_size, stride, verbose)
    implicit none
    class(avgpool2d_layer_type), intent(inout) :: this
    integer, dimension(2), intent(in) :: pool_size
    integer, dimension(2), intent(in) :: stride
    integer, optional, intent(in) :: verbose


    this%name = "avgpool2d"
    this%type = "pool"
    this%input_rank = 3
    allocate( &
         this%pool(this%input_rank-1), &
         this%strd(this%input_rank-1) &
    )
    this%pool = pool_size
    this%strd = stride

  end subroutine set_hyperparams_avgpool2d
!!!#############################################################################


!!!#############################################################################
!!! initialise layer
!!!#############################################################################
  subroutine init_avgpool2d(this, input_shape, batch_size, verbose)
    implicit none
    class(avgpool2d_layer_type), intent(inout) :: this
    integer, dimension(:), intent(in) :: input_shape
    integer, optional, intent(in) :: batch_size
    integer, optional, intent(in) :: verbose

    integer :: verbose_ = 0


    !!--------------------------------------------------------------------------
    !! initialise optional arguments
    !!--------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose
    if(present(batch_size)) this%batch_size = batch_size


    !!--------------------------------------------------------------------------
    !! initialise input shape
    !!--------------------------------------------------------------------------
    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)


    !!-----------------------------------------------------------------------
    !! set up number of channels, width, height
    !!-----------------------------------------------------------------------
    this%num_channels = this%input_shape(3)
    if(allocated(this%output))then
       if(this%output%allocated) call this%output%deallocate()
    end if
    this%output = array4d_type()
    this%output%shape(3) = this%input_shape(3)
    this%output%shape(:2) = &
         floor( (this%input_shape(:2) - this%pool)/real(this%strd)) + 1
    

    !!--------------------------------------------------------------------------
    !! initialise batch size-dependent arrays
    !!--------------------------------------------------------------------------
    if(this%batch_size.gt.0) call this%set_batch_size(this%batch_size)

  end subroutine init_avgpool2d
!!!#############################################################################


!!!#############################################################################
!!! set batch size
!!!#############################################################################
  subroutine set_batch_size_avgpool2d(this, batch_size, verbose)
    implicit none
    class(avgpool2d_layer_type), intent(inout) :: this
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
       if(.not.allocated(this%output)) this%output = array4d_type()
       if(this%output%allocated) call this%output%deallocate(keep_shape=.true.)
       call this%output%allocate( array_shape = [ &
            this%output%shape(1), &
            this%output%shape(2), this%num_channels, &
            this%batch_size ], &
            source=0._real32 &
       )
       if(.not.allocated(this%di)) this%di = array4d_type()
       if(this%di%allocated) call this%di%deallocate()
       call this%di%allocate( array_shape = [ &
            this%input_shape(1), &
            this%input_shape(2), &
            this%input_shape(3), &
            this%batch_size ], &
            source=0._real32 &
       )
    end if

  end subroutine set_batch_size_avgpool2d
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! print layer to file
!!!#############################################################################
  subroutine print_avgpool2d(this, file)
    implicit none
    class(avgpool2d_layer_type), intent(in) :: this
    character(*), intent(in) :: file

    integer :: unit

    !! open file with new unit
    !!--------------------------------------------------------------------------
    open(newunit=unit, file=trim(file), access='append')

    !! write convolution initial parameters
    !!--------------------------------------------------------------------------
    write(unit,'("AVGPOOL2D")')
    write(unit,'(3X,"INPUT_SHAPE = ",3(1X,I0))') this%input_shape
    if(all(this%pool.eq.this%pool(1)))then
       write(unit,'(3X,"POOL_SIZE =",1X,I0)') this%pool(1)
    else
       write(unit,'(3X,"POOL_SIZE =",2(1X,I0))') this%pool
    end if
    if(all(this%strd.eq.this%strd(1)))then
       write(unit,'(3X,"STRIDE =",1X,I0)') this%strd(1)
    else
       write(unit,'(3X,"STRIDE =",2(1X,I0))') this%strd
    end if
    write(unit,'("END AVGPOOL2D")')

    !! close unit
    !!--------------------------------------------------------------------------
    close(unit)

  end subroutine print_avgpool2d
!!!#############################################################################


!!!#############################################################################
!!! read layer from file
!!!#############################################################################
  subroutine read_avgpool2d(this, unit, verbose)
    use infile_tools, only: assign_val, assign_vec
    use misc, only: to_lower, icount
    implicit none
    class(avgpool2d_layer_type), intent(inout) :: this
    integer, intent(in) :: unit
    integer, optional, intent(in) :: verbose

    integer :: verbose_ = 0
    integer :: stat
    integer :: itmp1
    integer, dimension(2) :: pool_size, stride
    integer, dimension(3) :: input_shape
    character(256) :: buffer, tag


    if(present(verbose)) verbose_ = verbose

    !! loop over tags in layer card
    tag_loop: do

       !! check for end of file
       read(unit,'(A)',iostat=stat) buffer
       if(stat.ne.0)then
          write(0,*) "ERROR: file encountered error (EoF?) before END AVGPOOL2D"
          stop "Exiting..."
       end if
       if(trim(adjustl(buffer)).eq."") cycle tag_loop

       !! check for end of convolution card
       if(trim(adjustl(buffer)).eq."END AVGPOOL2D")then
          backspace(unit)
          exit tag_loop
       end if

       tag=trim(adjustl(buffer))
       if(scan(buffer,"=").ne.0) tag=trim(tag(:scan(tag,"=")-1))

       !! read parameters from save file
       select case(trim(tag))
       case("INPUT_SHAPE")
          call assign_vec(buffer, input_shape, itmp1)
       case("POOL_SIZE")
          call assign_vec(buffer, pool_size, itmp1)
       case("STRIDE")
          call assign_vec(buffer, stride, itmp1)
       case default
          !! don't look for "e" due to scientific notation of numbers
          !! ... i.e. exponent (E+00)
          if(scan(to_lower(trim(adjustl(buffer))),&
               'abcdfghijklmnopqrstuvwxyz').eq.0)then
             cycle tag_loop
          elseif(tag(:3).eq.'END')then
             cycle tag_loop
          end if
          stop "Unrecognised line in input file: "//trim(adjustl(buffer))
       end select
    end do tag_loop

    !! set transfer activation function
    call this%set_hyperparams(pool_size=pool_size, stride=stride)
    call this%init(input_shape = input_shape)

    !! check for end of layer card
    read(unit,'(A)') buffer
    if(trim(adjustl(buffer)).ne."END AVGPOOL2D")then
       write(*,*) trim(adjustl(buffer))
       stop "ERROR: END AVGPOOL2D not where expected"
    end if

  end subroutine read_avgpool2d
!!!#############################################################################


!!!#############################################################################
!!! read layer from file and return layer
!!!#############################################################################
  function read_avgpool2d_layer(unit, verbose) result(layer)
    implicit none
    integer, intent(in) :: unit
    integer, optional, intent(in) :: verbose
    class(avgpool2d_layer_type), allocatable :: layer

    integer :: verbose_ = 0


    if(present(verbose)) verbose_ = verbose
    call layer%read(unit, verbose=verbose_)

  end function read_avgpool2d_layer
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! forward propagation
!!!#############################################################################
  pure subroutine forward_4d(this, input)
    implicit none
    class(avgpool2d_layer_type), intent(inout) :: this
    real(real32), dimension( &
         this%input_shape(1), &
         this%input_shape(2), &
         this%num_channels, &
         this%batch_size), &
         intent(in) :: input

    integer :: i, j, m, s
    integer, dimension(2) :: stride_idx

    
    select type(output => this%output)
    type is (array4d_type)
       !! perform the pooling operation
       do concurrent(&
            s = 1:this%batch_size, &
            m = 1:this%num_channels, &
            j = 1:this%output%shape(2), &
            i = 1:this%output%shape(1))
#if defined(GFORTRAN)
          stride_idx = ([i,j] - 1) * this%strd + 1
#else
          stride_idx(1) = (i-1) * this%strd(1) + 1
          stride_idx(2) = (j-1) * this%strd(2) + 1
#endif
          output%val(i, j, m, s) = sum(&
               input( &
               stride_idx(1):stride_idx(1)+this%pool(1)-1, &
               stride_idx(2):stride_idx(2)+this%pool(2)-1, m, s)) / &
               product(this%pool)
       end do
    end select

  end subroutine forward_4d
!!!#############################################################################


!!!#############################################################################
!!! backward propagation
!!!#############################################################################
  pure subroutine backward_4d(this, input, gradient)
    implicit none
    class(avgpool2d_layer_type), intent(inout) :: this
    real(real32), dimension( &
         this%input_shape(1), &
         this%input_shape(2), &
         this%num_channels, &
         this%batch_size), &
         intent(in) :: input
    real(real32), &
         dimension(&
         this%output%shape(1), &
         this%output%shape(2), &
         this%num_channels, &
         this%batch_size), &
         intent(in) :: gradient

    integer :: i, j, m, s
    integer, dimension(2) :: stride_idx


    select type(di => this%di)
    type is (array4d_type)
       di%val = 0._real32
       !! compute gradients for input feature map
       do concurrent( &
            s = 1:this%batch_size, &
            m = 1:this%num_channels, &
            j = 1:this%output%shape(2), &
            i = 1:this%output%shape(1))
#if defined(GFORTRAN)
          stride_idx = ([i,j] - 1) * this%strd
#else
          stride_idx(1) = (i-1) * this%strd(1)
          stride_idx(2) = (j-1) * this%strd(2)
#endif
          !! compute gradients for input feature map
          di%val( &
               stride_idx(1)+1:stride_idx(1)+this%pool(1), &
               stride_idx(2)+1:stride_idx(2)+this%pool(2), m, s) = &
               di%val( &
               stride_idx(1)+1:stride_idx(1)+this%pool(1), &
               stride_idx(2)+1:stride_idx(2)+this%pool(2), m, s) + &
               gradient(i, j, m, s) / product(this%pool)
       end do
    end select

  end subroutine backward_4d
!!!#############################################################################

end module avgpool2d_layer
!!!#############################################################################
