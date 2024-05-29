!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module contains implementation of a 1D maxpooling layer
!!!#############################################################################
module maxpool1d_layer
  use constants, only: real12
  use base_layer, only: pool_layer_type
  implicit none
  
  
  type, extends(pool_layer_type) :: maxpool1d_layer_type
     real(real12), allocatable, dimension(:,:,:) :: output
     real(real12), allocatable, dimension(:,:,:) :: di ! gradient of input (i.e. delta)
   contains
     procedure, pass(this) :: get_output => get_output_maxpool1d
     procedure, pass(this) :: init => init_maxpool1d
     procedure, pass(this) :: set_batch_size => set_batch_size_maxpool1d
     procedure, pass(this) :: print => print_maxpool1d
     procedure, pass(this) :: forward  => forward_rank
     procedure, pass(this) :: backward => backward_rank
     procedure, private, pass(this) :: forward_3d
     procedure, private, pass(this) :: backward_3d
  end type maxpool1d_layer_type

  
  interface maxpool1d_layer_type
     module function layer_setup( &
          input_shape, batch_size, &
          pool_size, stride) result(layer)
       integer, dimension(:), optional, intent(in) :: input_shape
       integer, optional, intent(in) :: batch_size 
       integer, dimension(..), optional, intent(in) :: pool_size
       integer, dimension(..), optional, intent(in) :: stride
       type(maxpool1d_layer_type) :: layer
     end function layer_setup
  end interface maxpool1d_layer_type


  private
  public :: maxpool1d_layer_type
  public :: read_maxpool1d_layer


contains

!!!#############################################################################
!!! get layer outputs
!!!#############################################################################
  pure subroutine get_output_maxpool1d(this, output)
    implicit none
    class(maxpool1d_layer_type), intent(in) :: this
    real(real12), allocatable, dimension(..), intent(out) :: output
  
    select rank(output)
    rank(1)
       output = reshape(this%output, [size(this%output)])
    rank(2)
       output = &
            reshape(this%output, [product(this%output_shape),this%batch_size])
    rank(3)
       output = this%output
    end select
  
  end subroutine get_output_maxpool1d
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! forward propagation assumed rank handler
!!!#############################################################################
  pure subroutine forward_rank(this, input)
    implicit none
    class(maxpool1d_layer_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: input

    select rank(input); rank(3)
       call forward_3d(this, input)
    end select
  end subroutine forward_rank
!!!#############################################################################


!!!#############################################################################
!!! backward propagation assumed rank handler
!!!#############################################################################
  pure subroutine backward_rank(this, input, gradient)
    implicit none
    class(maxpool1d_layer_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: input
    real(real12), dimension(..), intent(in) :: gradient

    select rank(input); rank(3)
    select rank(gradient); rank(3)
      call backward_3d(this, input, gradient)
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
       pool_size, stride) result(layer)
    implicit none
    integer, dimension(:), optional, intent(in) :: input_shape
    integer, optional, intent(in) :: batch_size 
    integer, dimension(..), optional, intent(in) :: pool_size
    integer, dimension(..), optional, intent(in) :: stride
    
    type(maxpool1d_layer_type) :: layer
#else
  module procedure layer_setup
    implicit none
#endif


    layer%name = "maxpool1d"
    layer%input_rank = 2
    allocate( &
         layer%pool(layer%input_rank-1), &
         layer%strd(layer%input_rank-1) )
    !!-----------------------------------------------------------------------
    !! set up pool size
    !!-----------------------------------------------------------------------
    if(present(pool_size))then
       select rank(pool_size)
       rank(0)
          layer%pool = pool_size
       rank(1)
          layer%pool = pool_size
       end select
    else
       layer%pool = 2
    end if


    !!-----------------------------------------------------------------------
    !! set up stride
    !!-----------------------------------------------------------------------
    if(present(stride))then
       select rank(stride)
       rank(0)
          layer%strd = stride
       rank(1)
          layer%strd = stride
          end if
       end select
    else
       layer%strd = 2
    end if


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
!!! initialise layer
!!!#############################################################################
  subroutine init_maxpool1d(this, input_shape, batch_size, verbose)
    implicit none
    class(maxpool1d_layer_type), intent(inout) :: this
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
    this%num_channels = this%input_shape(2)
    allocate(this%output_shape(2))
    this%output_shape(2) = this%input_shape(2)
    this%output_shape(1)) = &
         floor( (this%input_shape(1) - this%pool)/real(this%strd)) + 1
    

    !!--------------------------------------------------------------------------
    !! initialise batch size-dependent arrays
    !!--------------------------------------------------------------------------
    if(this%batch_size.gt.0) call this%set_batch_size(this%batch_size)

  end subroutine init_maxpool1d
!!!#############################################################################


!!!#############################################################################
!!! set batch size
!!!#############################################################################
  subroutine set_batch_size_maxpool1d(this, batch_size, verbose)
   implicit none
   class(maxpool1d_layer_type), intent(inout) :: this
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
      allocate(this%output( &
           this%output_shape(1), this%num_channels, &
           this%batch_size), &
           source=0._real12)
      if(allocated(this%di)) deallocate(this%di)
      allocate(this%di( &
           this%input_shape(1), &
           this%input_shape(2), &
           this%batch_size), &
           source=0._real12)
   end if

 end subroutine set_batch_size_maxpool1d
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! print layer to file
!!!#############################################################################
  subroutine print_maxpool1d(this, file)
    implicit none
    class(maxpool1d_layer_type), intent(in) :: this
    character(*), intent(in) :: file

    integer :: unit

    !! open file with new unit
    !!--------------------------------------------------------------------------
    open(newunit=unit, file=trim(file), access='append')

    !! write convolution initial parameters
    !!--------------------------------------------------------------------------
    write(unit,'("MAXPOOL1D")')
    write(unit,'(3X,"INPUT_SHAPE = ",3(1X,I0))') this%input_shape
    write(unit,'(3X,"POOL_SIZE =",1X,I0)') this%pool(1)
    write(unit,'(3X,"STRIDE =",1X,I0)') this%strd(1)
    write(unit,'("END MAXPOOL1D")')

    !! close unit
    !!--------------------------------------------------------------------------
    close(unit)

  end subroutine print_maxpool1d
!!!#############################################################################


!!!#############################################################################
!!! read layer from file
!!!#############################################################################
  function read_maxpool1d_layer(unit) result(layer)
   use infile_tools, only: assign_val, assign_vec
   use misc, only: to_lower, icount
   implicit none
   integer, intent(in) :: unit

   class(maxpool1d_layer_type), allocatable :: layer

   integer :: stat
   integer :: itmp1
   integer, dimension(1) :: pool_size, stride
   integer, dimension(2) :: input_shape
   character(256) :: buffer, tag


   !! loop over tags in layer card
   tag_loop: do

      !! check for end of file
      read(unit,'(A)',iostat=stat) buffer
      if(stat.ne.0)then
         write(0,*) "ERROR: file encountered error (EoF?) before END MAXPOOL1D"
         stop "Exiting..."
      end if
      if(trim(adjustl(buffer)).eq."") cycle tag_loop

      !! check for end of convolution card
      if(trim(adjustl(buffer)).eq."END MAXPOOL1D")then
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

   layer = maxpool1d_layer_type( input_shape=input_shape, &
        pool_size = pool_size, stride = stride &
        )

   !! check for end of layer card
   read(unit,'(A)') buffer
   if(trim(adjustl(buffer)).ne."END MAXPOOL1D")then
      write(*,*) trim(adjustl(buffer))
      stop "ERROR: END MAXPOOL1D not where expected"
   end if

  end function read_maxpool1d_layer
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! forward propagation
!!!#############################################################################
  pure subroutine forward_3d(this, input)
    implicit none
    class(maxpool1d_layer_type), intent(inout) :: this
    real(real12), dimension( &
         this%input_shape(1), &
         this%num_channels, &
         this%batch_size), &
         intent(in) :: input

    integer :: i, m, s
    integer :: stride_idx

    
    this%output = 0._real12
    !! perform the pooling operation
    do concurrent(&
         s = 1:this%batch_size, &
         m = 1:this%num_channels, &
         i = 1:this%output_shape(1))
       stride_idx = (i-1) * this%strd(1) + 1
       this%output(i, m, s) = maxval(&
            input( &
            stride_idx:stride_idx+this%pool(1)-1, m, s))
    end do

  end subroutine forward_3d
!!!#############################################################################


!!!#############################################################################
!!! backward propagation
!!!#############################################################################
  pure subroutine backward_3d(this, input, gradient)
    implicit none
    class(maxpool1d_layer_type), intent(inout) :: this
    real(real12), dimension( &
         this%input_shape(1), &
         this%num_channels, &
         this%batch_size), &
         intent(in) :: input
    real(real12), &
         dimension(&
         this%output_shape(1), &
         this%num_channels, &
         this%batch_size), &
         intent(in) :: gradient

    integer :: i, m, s
    integer :: stride_idx, max_idx


    this%di = 0._real12
    !! compute gradients for input feature map
    do concurrent( &
         s = 1:this%batch_size, &
         m = 1:this%num_channels, &
         i = 1:this%output_shape(1))
       stride_idx = (i-1) * this%strd(1)
       !! find the index of the maximum value in the corresponding pooling window
       max_idx = maxloc(input( &
            stride_idx+1:stride_idx+this%pool(1), m, s), dim=1)

       !! compute gradients for input feature map
       this%di( &
            stride_idx+max_idx, m, s) = &
            this%di( &
            stride_idx+max_idx, m, s) + gradient(i, m, s)

    end do

  end subroutine backward_3d
!!!#############################################################################

end module maxpool1d_layer
!!!#############################################################################
