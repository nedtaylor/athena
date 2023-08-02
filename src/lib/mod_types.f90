!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module custom_types
  use constants, only: real12
  implicit none

!!!------------------------------------------------------------------------
!!! gradient clipping type
!!!------------------------------------------------------------------------
  type clip_type
     logical :: l_min_max, l_norm
     real(real12) :: min, max, norm
  end type clip_type


!!!------------------------------------------------------------------------
!!! learning parameter type
!!!------------------------------------------------------------------------
!!! MAKE THIS AN ABSTRACT TYPE WITH EXTENDED DERIVED TYPES FOR ADAM AND MOMENTUM
!!! THEN MAKE REGULARISATION A SUBTYPE
  type learning_parameters_type
     character(:), allocatable :: method
     !! reduce learning rate on plateau parameters
     !integer :: wait = 0
     !integer :: patience = 0
     !real(real12) :: factor = 0._real12
     !real(real12) :: min_learning_rate = 0._real12
     !! momentum parameters
     real(real12) :: momentum = 0._real12  ! fraction of momentum based learning
     !! step decay parameters
     !real(real12) :: decay_rate = 0._real12
     !real(real12) :: decay_steps = 0._real12
     !! adam optimiser parameters
     real(real12) :: beta1 = 0._real12
     real(real12) :: beta2 = 0._real12
     real(real12) :: epsilon = 0._real12
     !real(real12) :: weight_decay  ! L2 regularisation on Adam (AdamW)
     logical :: regularise = .false.
     character(:), allocatable :: regularisation
     real(real12) :: l1 = 0._real12
     real(real12) :: l2 = 0._real12
  end type learning_parameters_type


!!!------------------------------------------------------------------------
!!! fully connected network layer type
!!!------------------------------------------------------------------------
  type network_type
     real(real12), allocatable, dimension(:,:) :: weight, weight_incr
  end type network_type


!!!------------------------------------------------------------------------
!!! convolution layer type
!!!------------------------------------------------------------------------
  type convolution_type
     integer :: kernel_size
     integer :: stride
     integer :: pad
     integer :: centre_width
     real(real12) :: delta
     real(real12) :: bias, bias_incr
     !! DO THE WEIGHTS NEED TO BE DIFFERENT PER INPUT CHANNEL?
     !! IF SO, 3 DIMENSIONS. IF NOT, 2 DIMENSIONS
     real(real12), allocatable, dimension(:,:) :: weight, weight_incr
  end type convolution_type


!!!------------------------------------------------------------------------
!!! activation (transfer) function base type
!!!------------------------------------------------------------------------
!!! EXAMPLE OF HOW THIS WORKS WAS MODIFIED FROM:
!!! https://en.wikibooks.org/wiki/Fortran/OOP_in_Fortran
!!! https://stackoverflow.com/questions/19391094/is-it-possible-to-implement-an-abstract-variable-inside-a-type-in-fortran-2003
!!! https://stackoverflow.com/questions/8612466/how-to-alias-a-function-name-in-fortran
!!! https://www.adt.unipd.it/corsi/Bianco/www.pcc.qub.ac.uk/tec/courses/f90/stu-notes/F90_notesMIF_12.html
  type, abstract :: activation_type
     !! memory leak as allocatable character goes out of bounds
     !character(:), allocatable :: name
     character(10) :: name
     real(real12) :: scale
     real(real12) :: threshold
   contains
     procedure (activation_function_1d), deferred, pass(this) :: activate_1d
     procedure (derivative_function_1d), deferred, pass(this) :: differentiate_1d
     procedure (activation_function_3d), deferred, pass(this) :: activate_3d
     procedure (derivative_function_3d), deferred, pass(this) :: differentiate_3d
     generic :: activate => activate_1d, activate_3d 
     generic :: differentiate => differentiate_1d, differentiate_3d
  end type activation_type
  

  !! interface for activation function
  !!-----------------------------------------------------------------------
  abstract interface
     pure function activation_function_1d(this, val) result(output)
       import activation_type, real12
       class(activation_type), intent(in) :: this
       real(real12), dimension(:), intent(in) :: val
       real(real12), dimension(size(val,1)) :: output
     end function activation_function_1d

     pure function activation_function_3d(this, val) result(output)
       import activation_type, real12
       class(activation_type), intent(in) :: this
       real(real12), dimension(:,:,:), intent(in) :: val
       real(real12), dimension(size(val,1),size(val,2),size(val,3)) :: output
     end function activation_function_3d
  end interface


  !! interface for derivative function
  !!-----------------------------------------------------------------------
  abstract interface
     pure function derivative_function_1d(this, val) result(output)
       import activation_type, real12
       class(activation_type), intent(in) :: this
       real(real12), dimension(:), intent(in) :: val
       real(real12), dimension(size(val,1)) :: output
     end function derivative_function_1d

     pure function derivative_function_3d(this, val) result(output)
       import activation_type, real12
       class(activation_type), intent(in) :: this
       real(real12), dimension(:,:,:), intent(in) :: val
       real(real12), dimension(size(val,1),size(val,2),size(val,3)) :: output
     end function derivative_function_3d
  end interface


!!!------------------------------------------------------------------------
!!! weights and biases initialiser base type
!!!------------------------------------------------------------------------
  type, abstract :: initialiser_type
   contains
     procedure (initialiser_subroutine), deferred, pass(this) :: initialise
  end type initialiser_type


  !! interface for initialiser function
  !!-----------------------------------------------------------------------
  abstract interface
     subroutine initialiser_subroutine(this, input, fan_in, fan_out)
       import initialiser_type, real12
       class(initialiser_type), intent(inout) :: this
       real(real12), dimension(..), intent(out) :: input
       integer, optional, intent(in) :: fan_in, fan_out
       real(real12) :: scale
     end subroutine initialiser_subroutine
  end interface



  private

  public :: clip_type
  public :: network_type
  public :: convolution_type
  public :: activation_type
  public :: learning_parameters_type
  public :: initialiser_type

end module custom_types
!!!#############################################################################
