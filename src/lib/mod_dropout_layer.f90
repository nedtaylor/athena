!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module contains implementation of a dropout layer
!!!#############################################################################
!!! Dropout reference: ...
!!! ... https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
!!!#############################################################################
module dropout_layer
  use constants, only: real32
  use base_layer, only: drop_layer_type
  use custom_types, only: array2d_type
  implicit none
  
  
  type, extends(drop_layer_type) :: dropout_layer_type
     !! num_masks              -- number of unique masks = number of samples in batch
     !! idx                    -- temp index of sample (doesn't need to be accurate)
     !! keep_prob              -- typical = 0.75-0.95
     integer :: idx = 0
     integer :: num_masks
     logical, allocatable, dimension(:,:) :: mask
    !  real(real32), allocatable, dimension(:,:) :: output
    !  real(real32), allocatable, dimension(:,:) :: di ! gradient of input (i.e. delta)
   contains
     procedure, pass(this) :: set_hyperparams => set_hyperparams_dropout
     procedure, pass(this) :: init => init_dropout
     procedure, pass(this) :: set_batch_size => set_batch_size_dropout
     procedure, pass(this) :: print => print_dropout
     procedure, pass(this) :: read => read_dropout
     procedure, pass(this) :: forward  => forward_rank
     procedure, pass(this) :: backward => backward_rank
     procedure, private, pass(this) :: forward_2d
     procedure, private, pass(this) :: backward_2d
     procedure, pass(this) :: generate_mask => generate_dropout_mask
  end type dropout_layer_type

  
  interface dropout_layer_type
     module function layer_setup( &
          rate, num_masks, &
          input_shape, batch_size) result(layer)
       integer, intent(in) :: num_masks
       real(real32), intent(in) :: rate
       integer, dimension(:), optional, intent(in) :: input_shape
       integer, optional, intent(in) :: batch_size
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
    real(real32), dimension(..), intent(in) :: input

    select rank(input); rank(2)
       call forward_2d(this, input)
    end select
  end subroutine forward_rank
!!!#############################################################################


!!!#############################################################################
!!! backward propagation assumed rank handler
!!!#############################################################################
  pure subroutine backward_rank(this, input, gradient)
    implicit none
    class(dropout_layer_type), intent(inout) :: this
    real(real32), dimension(..), intent(in) :: input
    real(real32), dimension(..), intent(in) :: gradient

    select rank(input); rank(2)
    select rank(gradient); rank(2)
      call backward_2d(this, input, gradient)
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
       input_shape, batch_size) result(layer)
    implicit none
    integer, intent(in) :: num_masks
    real(real32), intent(in) :: rate
    integer, dimension(:), optional, intent(in) :: input_shape
    integer, optional, intent(in) :: batch_size
    
    type(dropout_layer_type) :: layer


    !!--------------------------------------------------------------------------
    !! initialise hyperparameters
    !!--------------------------------------------------------------------------
    call layer%set_hyperparams(rate, num_masks)


    !!--------------------------------------------------------------------------
    !! initialise batch size
    !!--------------------------------------------------------------------------
    if(present(batch_size)) layer%batch_size = batch_size


    !!--------------------------------------------------------------------------
    !! initialise layer shape
    !!--------------------------------------------------------------------------
    if(present(input_shape)) call layer%init(input_shape=input_shape)

  end function layer_setup
!!!#############################################################################


!!!#############################################################################
!!! set hyperparameters
!!!#############################################################################
  pure subroutine set_hyperparams_dropout(this, rate, num_masks)
    implicit none
    class(dropout_layer_type), intent(inout) :: this
    real(real32), intent(in) :: rate
    integer, intent(in) :: num_masks

    this%name = "dropout"
    this%type = "drop"
    this%input_rank = 1

    this%num_masks = num_masks
    this%rate = rate

  end subroutine set_hyperparams_dropout
!!!#############################################################################


!!!#############################################################################
!!! initialise layer
!!!#############################################################################
  subroutine init_dropout(this, input_shape, batch_size, verbose)
    implicit none
    class(dropout_layer_type), intent(inout) :: this
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
    this%output%shape = this%input_shape


    !!-----------------------------------------------------------------------
    !! allocate mask
    !!-----------------------------------------------------------------------
    allocate(this%mask(this%input_shape(1), this%num_masks), source=.true.)


    !!-----------------------------------------------------------------------
    !! generate mask
    !!-----------------------------------------------------------------------
    call this%generate_mask()
    

    !!--------------------------------------------------------------------------
    !! initialise batch size-dependent arrays
    !!--------------------------------------------------------------------------
    if(this%batch_size.gt.0) call this%set_batch_size(this%batch_size)

  end subroutine init_dropout
!!!#############################################################################


!!!#############################################################################
!!! set batch size
!!!#############################################################################
  subroutine set_batch_size_dropout(this, batch_size, verbose)
    implicit none
    class(dropout_layer_type), intent(inout) :: this
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
       if(this%output%allocated) call this%output%deallocate()

       this%output = array2d_type()

       call this%output%allocate( array_shape = [ &
            this%output%shape(1), &
            this%batch_size ], source=0._real32 &
       )
       if(this%di%allocated) call this%di%deallocate()
       this%di = array2d_type()
       call this%di%allocate( source=this%output )
    end if
 
  end subroutine set_batch_size_dropout
 !!!#############################################################################
 

!!!#############################################################################
!!! generate masks
!!!#############################################################################
  subroutine generate_dropout_mask(this)
    implicit none
    class(dropout_layer_type), intent(inout) :: this
    real(real32), allocatable, dimension(:,:) :: mask_real

    
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
  subroutine read_dropout(this, unit, verbose)
    use infile_tools, only: assign_val, assign_vec
    use misc, only: to_lower, icount
    implicit none
    class(dropout_layer_type), intent(inout) :: this
    integer, intent(in) :: unit
    integer, optional, intent(in) :: verbose

    integer :: verbose_ = 0
    integer :: stat
    integer :: itmp1
    integer :: num_masks
    real(real32) :: rate
    integer, dimension(3) :: input_shape
    character(256) :: buffer, tag


    if(present(verbose)) verbose_ = verbose

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

    call this%set_hyperparams(rate = rate, num_masks = num_masks)
    call this%init(input_shape = input_shape)

    !! check for end of layer card
    read(unit,'(A)') buffer
    if(trim(adjustl(buffer)).ne."END DROPOUT")then
       write(*,*) trim(adjustl(buffer))
       stop "ERROR: END DROPOUT not where expected"
    end if

  end subroutine read_dropout
!!!#############################################################################


!!!#############################################################################
!!! read layer from file and return layer
!!!#############################################################################
  function read_dropout_layer(unit, verbose) result(layer)
    implicit none
    integer, intent(in) :: unit
    integer, optional, intent(in) :: verbose
    class(dropout_layer_type), allocatable :: layer

    integer :: verbose_ = 0


    if(present(verbose)) verbose_ = verbose
    call layer%read(unit, verbose=verbose_)

  end function read_dropout_layer
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! forward propagation
!!!#############################################################################
  pure subroutine forward_2d(this, input)
    implicit none
    class(dropout_layer_type), intent(inout) :: this
    real(real32), dimension( &
         this%input_shape(1), this%batch_size), &
         intent(in) :: input
    
    integer :: s

    
    select type(output => this%output)
    type is (array2d_type)
       select case(this%inference)
       case(.true.)
          !! do not perform the drop operation
          output%val = input * ( 1._real32 - this%rate )
       case default
          !! perform the drop operation
          this%idx = this%idx + 1
          do concurrent(s=1:this%batch_size)
              output%val(:,s) = merge( &
                  input(:,s), 0._real32, &
                  this%mask(:,this%idx)) / &
                  ( 1._real32 - this%rate )
          end do
       end select
    end select

  end subroutine forward_2d
!!!#############################################################################


!!!#############################################################################
!!! backward propagation
!!!#############################################################################
  pure subroutine backward_2d(this, input, gradient)
    implicit none
    class(dropout_layer_type), intent(inout) :: this
    real(real32), dimension( &
         this%input_shape(1), this%batch_size), &
         intent(in) :: input
    real(real32), &
         dimension(this%output%shape(1), this%batch_size), &
         intent(in) :: gradient


    !! compute gradients for input feature map
    select type(di => this%di)
    type is (array2d_type)
       di%val(:,:) = gradient(:,:)
    end select

  end subroutine backward_2d
!!!#############################################################################

end module dropout_layer
!!!#############################################################################
