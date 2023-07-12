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
  type learning_parameters_type
     character(:), allocatable :: method
     !! reduce learning rate on plateau parameters
     integer :: wait = 0
     integer :: patience = 0
     real(real12) :: factor = 0._real12
     real(real12) :: min_learning_rate = 0._real12
     !! momentum parameters
     real(real12) :: momentum = 0._real12  ! fraction of momentum based learning
     !! step decay parameters
     real(real12) :: decay_rate = 0._real12
     real(real12) :: decay_steps = 0._real12
     !! adam optimiser parameters
     real(real12) :: beta1 = 0._real12
     real(real12) :: beta2 = 0._real12
     real(real12) :: epsilon = 0._real12
     !real(real12) :: weight_decay  ! L2 regularisation on Adam (AdamW)
  end type learning_parameters_type


!!!------------------------------------------------------------------------
!!! neural network neuron type
!!!------------------------------------------------------------------------
  type neuron_type
     real(real12) :: output
     real(real12) :: delta, delta_batch
     real(real12), allocatable, dimension(:) :: weight, weight_incr
  end type neuron_type


!!!------------------------------------------------------------------------
!!! fully connected network layer type
!!!------------------------------------------------------------------------
  type network_type
     type(neuron_type), allocatable, dimension(:) :: neuron
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
     real(real12) :: bias
     !! DO THE WEIGHTS NEED TO BE DIFFERENT PER INPUT CHANNEL?
     !! IF SO, 3 DIMENSIONS. IF NOT, 2 DIMENSIONS
     real(real12), allocatable, dimension(:,:) :: weight, weight_incr
     !real(real12), allocatable, dimension(:,:,:) :: output
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
     real(real12) :: scale
     real(real12) :: threshold
   contains
     procedure (activation_function), deferred :: activate
     procedure (derivative_function), deferred :: differentiate
  end type activation_type
  

  !! interface for activation function
  !!-----------------------------------------------------------------------
  abstract interface
     function activation_function(this, val) result(output)
       import activation_type, real12
       class(activation_type), intent(in) :: this
       real(real12), intent(in) :: val
       real(real12) :: output
     end function activation_function
  end interface


  !! interface for derivative function
  !!-----------------------------------------------------------------------
  abstract interface
     function derivative_function(this, val) result(output)
       import activation_type, real12
       class(activation_type), intent(in) :: this
       real(real12), intent(in) :: val
       real(real12) :: output
     end function derivative_function
  end interface


!!!------------------------------------------------------------------------
!!! weights and biases initialiser base type
!!!------------------------------------------------------------------------
  type, abstract :: initialiser_type
   contains
     procedure (initialiser_subroutine), deferred :: initialise
  end type initialiser_type


  !! interface for initialiser function
  !!-----------------------------------------------------------------------
  abstract interface
     subroutine initialiser_subroutine(this, input, fan_in, fan_out)
       import initialiser_type, real12
       class(initialiser_type), intent(inout) :: this
       real(real12), dimension(..), intent(out) :: input
       integer, optional, intent(in) :: fan_in, fan_out
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
