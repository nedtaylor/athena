!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module dropout_layer
  use constants, only: real12
  use base_layer, only: drop_layer_type
  implicit none
  
  
  type, extends(drop_layer_type) :: dropout_layer_type
     !! num_masks              -- number of unique masks = number of samples in batch
     !! idx                    -- temp index of sample (doesn't need to be accurate)
     !! keep_prob              -- typical = 0.75-0.95
     !! rate = 1 - keep_prob   -- typical = 0.05-0.25
     integer :: idx = 0
     integer :: num_masks
     real(real12) :: rate
     logical, allocatable, dimension(:,:) :: mask
     real(real12), allocatable, dimension(:) :: output
     real(real12), allocatable, dimension(:) :: di ! gradient of input (i.e. delta)
   contains
     procedure, pass(this) :: init => init_dropout
     procedure, pass(this) :: print => print_dropout
     procedure, pass(this) :: forward  => forward_rank
     procedure, pass(this) :: backward => backward_rank
     procedure, private, pass(this) :: forward_1d
     procedure, private, pass(this) :: backward_1d
     procedure, pass(this) :: generate_mask => generate_dropout_mask
  end type dropout_layer_type

  
  interface dropout_layer_type
     module function layer_setup( &
          rate, num_masks, &
          input_shape) result(layer)
       integer, intent(in) :: num_masks
       real(real12), intent(in) :: rate
       integer, dimension(:), optional, intent(in) :: input_shape
       type(dropout_layer_type) :: layer
     end function layer_setup
  end interface dropout_layer_type


  private
  public :: dropout_layer_type
  public :: read_dropout_layer


contains

!!!#############################################################################
!!! forward propagation assumed rank handler
!!!#############################################################################
  pure subroutine forward_rank(this, input)
    implicit none
    class(dropout_layer_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: input

    select rank(input); rank(1)
       call forward_1d(this, input)
    end select
  end subroutine forward_rank
!!!#############################################################################


!!!#############################################################################
!!! backward propagation assumed rank handler
!!!#############################################################################
  pure subroutine backward_rank(this, input, gradient)
    implicit none
    class(dropout_layer_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: input
    real(real12), dimension(..), intent(in) :: gradient

    select rank(input); rank(1)
    select rank(gradient); rank(1)
      call backward_1d(this, input, gradient)
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
  module function layer_setup( &
       rate, num_masks, &
       input_shape) result(layer)
    implicit none
    integer, intent(in) :: num_masks
    real(real12), intent(in) :: rate
    integer, dimension(:), optional, intent(in) :: input_shape
    
    type(dropout_layer_type) :: layer


    layer%num_masks = num_masks
    layer%rate = rate


    !!--------------------------------------------------------------------------
    !! initialise layer shape
    !!--------------------------------------------------------------------------
    if(present(input_shape)) call layer%init(input_shape=input_shape)

  end function layer_setup
!!!#############################################################################


!!!#############################################################################
!!! initialise layer
!!!#############################################################################
  subroutine init_dropout(this, input_shape, verbose)
    implicit none
    class(dropout_layer_type), intent(inout) :: this
    integer, dimension(:), intent(in) :: input_shape
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


    !!--------------------------------------------------------------------------
    !! initialise input shape
    !!--------------------------------------------------------------------------
    if(size(input_shape,dim=1).eq.1)then
       this%input_shape = input_shape
    else
       stop "ERROR: invalid size of input_shape in dropout, expected (1)"
    end if


    !!-----------------------------------------------------------------------
    !! set up number of channels, width, height
    !!-----------------------------------------------------------------------
    this%output_shape = input_shape
    

    !!-----------------------------------------------------------------------
    !! allocate output and gradients
    !!-----------------------------------------------------------------------
    allocate(this%output(this%output_shape(1)), source=0._real12)
    allocate(this%di(input_shape(1)), source=0._real12)


    !!-----------------------------------------------------------------------
    !! set gamma
    !! ... original paper: https://doi.org/10.1145/3474085.3475302
    !!-----------------------------------------------------------------------
    !! original paper uses keep_prob, we use drop_rate
    !! drop_rate = 1 - keep_prob
    allocate(this%mask(input_shape(1), this%num_masks), source=.true.)


    !!-----------------------------------------------------------------------
    !! generate mask
    !!-----------------------------------------------------------------------
    call this%generate_mask()

  end subroutine init_dropout
!!!#############################################################################


!!!#############################################################################
!!! generate masks
!!!#############################################################################
  subroutine generate_dropout_mask(this)
    implicit none
    class(dropout_layer_type), intent(inout) :: this
    real(real12), allocatable, dimension(:,:) :: mask_real

    
    !! generate masks
    allocate(mask_real(size(this%mask,1), size(this%mask,2)))
    call random_number(mask_real)  ! Generate random values in [0..1]
    this%mask = mask_real > this%rate

    this%idx = 0

  end subroutine generate_dropout_mask
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! print layer to file
!!!#############################################################################
  subroutine print_dropout(this, file)
    implicit none
    class(dropout_layer_type), intent(in) :: this
    character(*), intent(in) :: file

    integer :: unit


    !! open file with new unit
    !!--------------------------------------------------------------------------
    open(newunit=unit, file=trim(file), access='append')

    !! write convolution initial parameters
    !!--------------------------------------------------------------------------
    write(unit,'("DROPOUT")')
    write(unit,'(3X,"INPUT_SHAPE = ",3(1X,I0))') this%input_shape
    write(unit,'(3X,"RATE = ",F0.9)') this%rate
    write(unit,'(3X,"NUM_MASKS = ",I0)') this%num_masks
    write(unit,'("END DROPOUT")')

    !! close unit
    !!--------------------------------------------------------------------------
    close(unit)

  end subroutine print_dropout
!!!#############################################################################


!!!#############################################################################
!!! read layer from file
!!!#############################################################################
  function read_dropout_layer(unit) result(layer)
   use infile_tools, only: assign_val, assign_vec
   use misc, only: to_lower, icount
   implicit none
   integer, intent(in) :: unit

   class(dropout_layer_type), allocatable :: layer

   integer :: stat
   integer :: itmp1
   integer :: num_masks
   real(real12) :: rate
   integer, dimension(3) :: input_shape
   character(256) :: buffer, tag


   !! loop over tags in layer card
   tag_loop: do

      !! check for end of file
      read(unit,'(A)',iostat=stat) buffer
      if(stat.ne.0)then
         write(0,*) "ERROR: file encountered error (EoF?) before END DROPOUT"
         stop "Exiting..."
      end if
      if(trim(adjustl(buffer)).eq."") cycle tag_loop

      !! check for end of convolution card
      if(trim(adjustl(buffer)).eq."END DROPOUT")then
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
      case("NUM_MASKS")
         call assign_val(buffer, num_masks, itmp1)
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

   layer = dropout_layer_type(rate = rate, num_masks = num_masks, &
        input_shape = input_shape)

   !! check for end of layer card
   read(unit,'(A)') buffer
   if(trim(adjustl(buffer)).ne."END DROPOUT")then
      write(*,*) trim(adjustl(buffer))
      stop "ERROR: END DROPOUT not where expected"
   end if

  end function read_dropout_layer
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! forward propagation
!!!#############################################################################
  pure subroutine forward_1d(this, input)
    implicit none
    class(dropout_layer_type), intent(inout) :: this
    real(real12), dimension(:), intent(in) :: input
    
    
    this%idx = this%idx + 1
    !! perform the drop operation
    this%output(:) = merge(input(:), 0._real12, this%mask(:,this%idx)) / &
       ( 1._real12 - this%rate )

  end subroutine forward_1d
!!!#############################################################################


!!!#############################################################################
!!! backward propagation
!!!#############################################################################
  pure subroutine backward_1d(this, input, gradient)
    implicit none
    class(dropout_layer_type), intent(inout) :: this
    real(real12), dimension(:), intent(in) :: input
    real(real12), &
         dimension(this%output_shape(1)), intent(in) :: gradient


    !! compute gradients for input feature map
    this%di(:) = gradient(:)

  end subroutine backward_1d
!!!#############################################################################

end module dropout_layer
!!!#############################################################################
