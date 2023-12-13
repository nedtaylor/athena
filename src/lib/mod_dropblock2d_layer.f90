!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module dropblock2d_layer
  use constants, only: real12
  use base_layer, only: drop_layer_type
  implicit none
  
  
  type, extends(drop_layer_type) :: dropblock2d_layer_type
     !! keep_prob              -- typical = 0.75-0.95
     !! block_size             -- width of block to drop (typical = 5)
     !! gamma                  -- number of activation units to drop
     integer :: block_size, half
     real(real12) :: gamma
     integer :: num_channels
     logical, allocatable, dimension(:,:) :: mask
     real(real12), allocatable, dimension(:,:,:,:) :: output
     real(real12), allocatable, dimension(:,:,:,:) :: di ! gradient of input (i.e. delta)
   contains
     procedure, pass(this) :: get_output => get_output_dropblock2d
     procedure, pass(this) :: init => init_dropblock2d
     procedure, pass(this) :: set_batch_size => set_batch_size_dropblock2d
     procedure, pass(this) :: print => print_dropblock2d
     procedure, pass(this) :: forward  => forward_rank
     procedure, pass(this) :: backward => backward_rank
     procedure, private, pass(this) :: forward_4d
     procedure, private, pass(this) :: backward_4d
     procedure, pass(this) :: generate_mask => generate_bernoulli_mask
  end type dropblock2d_layer_type

  
  interface dropblock2d_layer_type
     module function layer_setup( &
          rate, block_size, &
          input_shape, batch_size) result(layer)
       real(real12), intent(in) :: rate
       integer, intent(in) :: block_size
       integer, dimension(:), optional, intent(in) :: input_shape
       integer, optional, intent(in) :: batch_size
       type(dropblock2d_layer_type) :: layer
     end function layer_setup
  end interface dropblock2d_layer_type


  private
  public :: dropblock2d_layer_type
  public :: read_dropblock2d_layer


contains

!!!#############################################################################
!!! get layer outputs
!!!#############################################################################
  pure subroutine get_output_dropblock2d(this, output)
    implicit none
    class(dropblock2d_layer_type), intent(in) :: this
    real(real12), allocatable, dimension(..), intent(out) :: output
  
    select rank(output)
    rank(1)
       output = reshape(this%output, [size(this%output)])
    rank(2)
       output = &
            reshape(this%output, [product(this%output_shape),this%batch_size])
    rank(4)
       output = this%output
    end select
  
  end subroutine get_output_dropblock2d
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! forward propagation assumed rank handler
!!!#############################################################################
  pure subroutine forward_rank(this, input)
    implicit none
    class(dropblock2d_layer_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: input

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
    class(dropblock2d_layer_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: input
    real(real12), dimension(..), intent(in) :: gradient

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
       rate, block_size, &
       input_shape, batch_size) result(layer)
    implicit none
    real(real12), intent(in) :: rate
    integer, intent(in) :: block_size
    integer, dimension(:), optional, intent(in) :: input_shape
    integer, optional, intent(in) :: batch_size
    
    type(dropblock2d_layer_type) :: layer
#else
  module procedure layer_setup
    implicit none
#endif


    layer%name = "dropblock2d"
    layer%input_rank = 3
    !!--------------------------------------------------------------------------
    !! initialise batch size
    !!--------------------------------------------------------------------------
    if(present(batch_size)) layer%batch_size = batch_size


    !!--------------------------------------------------------------------------
    !! initialise layer rate and block size
    !!--------------------------------------------------------------------------
    layer%rate = rate
    layer%block_size = block_size
    layer%half = (layer%block_size-1)/2


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
  subroutine init_dropblock2d(this, input_shape, batch_size, verbose)
    implicit none
    class(dropblock2d_layer_type), intent(inout) :: this
    integer, dimension(:), intent(in) :: input_shape
    integer, optional, intent(in) :: batch_size
    integer, optional, intent(in) :: verbose

    integer :: t_verb


    !!--------------------------------------------------------------------------
    !! initialise optional arguments
    !!--------------------------------------------------------------------------
    if(present(verbose))then
       t_verb = verbose
    else
       t_verb = 0
    end if
    if(present(batch_size)) this%batch_size = batch_size


    !!--------------------------------------------------------------------------
    !! initialise input shape
    !!--------------------------------------------------------------------------
    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)


    !!-----------------------------------------------------------------------
    !! set up number of channels, width, height
    !!-----------------------------------------------------------------------
    this%num_channels = this%input_shape(3)
    this%output_shape = this%input_shape


    !!-----------------------------------------------------------------------
    !! set gamma
    !! ... original paper: https://doi.org/10.1145/3474085.3475302
    !!-----------------------------------------------------------------------
    !! original paper uses keep_prob, we use drop_rate
    !! drop_rate = 1 - keep_prob
    this%gamma = ( this%rate/this%block_size**2._real12 ) * &
         this%input_shape(1) / &
              (this%input_shape(1) - this%block_size + 1._real12) * &
         this%input_shape(2) / &
              (this%input_shape(2) - this%block_size + 1._real12)
    allocate(this%mask( &
         this%input_shape(1), &
         this%input_shape(2)), source=.true.)


    !!-----------------------------------------------------------------------
    !! generate mask
    !!-----------------------------------------------------------------------
    call this%generate_mask()
    

    !!--------------------------------------------------------------------------
    !! initialise batch size-dependent arrays
    !!--------------------------------------------------------------------------
    if(this%batch_size.gt.0) call this%set_batch_size(this%batch_size)

  end subroutine init_dropblock2d
!!!#############################################################################


!!!#############################################################################
!!! set batch size
!!!#############################################################################
  subroutine set_batch_size_dropblock2d(this, batch_size, verbose)
    implicit none
    class(dropblock2d_layer_type), intent(inout) :: this
    integer, intent(in) :: batch_size
    integer, optional, intent(in) :: verbose
 
    integer :: t_verb
 
 
    !!--------------------------------------------------------------------------
    !! initialise optional arguments
    !!--------------------------------------------------------------------------
    if(present(verbose))then
       t_verb = verbose
    else
       t_verb = 0
    end if
    this%batch_size = batch_size
 
 
    !!--------------------------------------------------------------------------
    !! allocate arrays
    !!--------------------------------------------------------------------------
    if(allocated(this%input_shape))then
       if(allocated(this%output)) deallocate(this%output)
       allocate(this%output( &
            this%output_shape(1), &
            this%output_shape(2), &
            this%num_channels, &
            this%batch_size), &
            source=0._real12)
       if(allocated(this%di)) deallocate(this%di)
       allocate(this%di, source=this%output)
    end if
 
  end subroutine set_batch_size_dropblock2d
 !!!#############################################################################
 

!!!#############################################################################
!!! generate bernoulli mask
!!!#############################################################################
  subroutine generate_bernoulli_mask(this)
    implicit none
    class(dropblock2d_layer_type), intent(inout) :: this

    real(real12), allocatable, dimension(:,:) :: mask_real
    integer :: i, j
    integer, dimension(2) :: ilim, jlim
    

    !! IF seed GIVEN, INITIALISE
    ! assume random number already seeded and don't need to again
    allocate(mask_real(size(this%mask,1), size(this%mask,2)))
    call random_number(mask_real)  ! Generate random values in [0..1]

    this%mask = .true. !1=keep

    !! Apply threshold to create binary mask
    do j = 1 + this%half, size(this%mask, dim=2) - this%half
       do i = 1 + this%half, size(this%mask, dim=1) - this%half
          if(mask_real(i, j).gt.this%gamma)then
             ilim(:) = [ &
                  max(i - this%half, lbound(this%mask,1)), &
                  min(i + this%half, ubound(this%mask,1)) ]
             jlim(:) = [ &
                  max(j - this%half, lbound(this%mask,2)), &
                  min(j + this%half, ubound(this%mask,2)) ]
             this%mask(ilim(1):ilim(2), jlim(1):jlim(2)) = .false. !0 = drop
          end if
       end do
    end do

  end subroutine generate_bernoulli_mask
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! print layer to file
!!!#############################################################################
  subroutine print_dropblock2d(this, file)
    implicit none
    class(dropblock2d_layer_type), intent(in) :: this
    character(*), intent(in) :: file

    integer :: unit


    !! open file with new unit
    !!--------------------------------------------------------------------------
    open(newunit=unit, file=trim(file), access='append')

    !! write convolution initial parameters
    !!--------------------------------------------------------------------------
    write(unit,'("DROPBLOCK2D")')
    write(unit,'(3X,"INPUT_SHAPE = ",3(1X,I0))') this%input_shape
    write(unit,'(3X,"RATE = ",F0.9)') this%rate
    write(unit,'(3X,"BLOCK_SIZE = ",I0)') this%block_size
    write(unit,'("END DROPBLOCK2D")')

    !! close unit
    !!--------------------------------------------------------------------------
    close(unit)

  end subroutine print_dropblock2d
!!!#############################################################################


!!!#############################################################################
!!! read layer from file
!!!#############################################################################
  function read_dropblock2d_layer(unit) result(layer)
   use infile_tools, only: assign_val, assign_vec
   use misc, only: to_lower, icount
   implicit none
   integer, intent(in) :: unit

   class(dropblock2d_layer_type), allocatable :: layer

   integer :: stat
   integer :: itmp1
   integer :: block_size
   real(real12) :: rate
   integer, dimension(3) :: input_shape
   character(256) :: buffer, tag


   !! loop over tags in layer card
   tag_loop: do

      !! check for end of file
      read(unit,'(A)',iostat=stat) buffer
      if(stat.ne.0)then
         write(0,*) "ERROR: file encountered error (EoF?) before END DROPBLOCK2D"
         stop "Exiting..."
      end if
      if(trim(adjustl(buffer)).eq."") cycle tag_loop

      !! check for end of convolution card
      if(trim(adjustl(buffer)).eq."END DROPBLOCK2D")then
         backspace(unit)
         exit tag_loop
      end if

      tag=trim(adjustl(buffer))
      if(scan(buffer,"=").ne.0) tag=trim(tag(:scan(tag,"=")-1))

      !! read parameters from save file
      select case(trim(tag))
      case("INPUT_SHAPE")
         call assign_vec(buffer, input_shape, itmp1)
      case("RATE")
         call assign_val(buffer, rate, itmp1)
      case("BLOCK_SIZE")
         call assign_val(buffer, block_size, itmp1)
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

   layer = dropblock2d_layer_type(rate = rate, block_size = block_size, &
        input_shape = input_shape)

   !! check for end of layer card
   read(unit,'(A)') buffer
   if(trim(adjustl(buffer)).ne."END DROPBLOCK2D")then
      write(*,*) trim(adjustl(buffer))
      stop "ERROR: END DROPBLOCK2D not where expected"
   end if

  end function read_dropblock2d_layer
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! forward propagation
!!!#############################################################################
  pure subroutine forward_4d(this, input)
    implicit none
    class(dropblock2d_layer_type), intent(inout) :: this
    real(real12), dimension(:,:,:,:), intent(in) :: input

    integer :: m, s


    select case(this%inference)
    case(.true.)
      !! do not perform drop operation
      this%output = input * ( 1._real12 - this%rate )
    case default
      !! perform the drop operation
      do concurrent(m = 1:this%num_channels, s = 1:this%batch_size)
         this%output(:,:,m,s) = merge(input(:,:,m,s), 0._real12, this%mask)
      end do
    end select

  end subroutine forward_4d
!!!#############################################################################


!!!#############################################################################
!!! backward propagation
!!!#############################################################################
  pure subroutine backward_4d(this, input, gradient)
    implicit none
    class(dropblock2d_layer_type), intent(inout) :: this
    real(real12), dimension(:,:,:,:), intent(in) :: input
    real(real12), &
         dimension(&
         this%output_shape(1), &
         this%output_shape(2), &
         this%num_channels, this%batch_size), &
         intent(in) :: gradient

    integer :: m, s


    !! compute gradients for input feature map
    do concurrent(m = 1:this%num_channels, s=1:this%batch_size)
       this%di(:,:,m,s) = merge(gradient(:,:,m,s), 0._real12, this%mask)
    end do

  end subroutine backward_4d
!!!#############################################################################

end module dropblock2d_layer
!!!#############################################################################
