!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module optimiser
  use constants, only: real12
  implicit none

!!!------------------------------------------------------------------------
!!! gradient clipping type
!!!------------------------------------------------------------------------
  type clip_type
     logical :: l_min_max = .false.
     logical :: l_norm    = .false.
     real(real12) :: min  =-huge(1._real12)
     real(real12) :: max  = huge(1._real12)
     real(real12) :: norm = huge(1._real12)
  end type clip_type

!!!-----------------------------------------------------------------------------
!!! learning parameter type
!!!-----------------------------------------------------------------------------
!!! MAKE THIS AN ABSTRACT TYPE WITH EXTENDED DERIVED TYPES FOR ADAM AND MOMENTUM
!!! THEN MAKE REGULARISATION A SUBTYPE
!!! NO, DON'T MAKE IT AN ABSTRACT TYPE, JUST MAKE IT A TYPE THAT HAS DERIVED TYPES FROM IT
  type optimiser_type
     character(:), allocatable :: method
     !! reduce learning rate on plateau parameters
     !integer :: wait = 0
     !integer :: patience = 0
     !real(real12) :: factor = 0._real12
     !real(real12) :: min_learning_rate = 0._real12
     real(real12) :: learning_rate
     integer :: iter
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
     type(clip_type) :: clip_dict
   contains
     procedure, pass(this) :: optimise
     procedure, pass(this) :: set_clip
     procedure, pass(this) :: clip => clip_gradients
     !procedure, private, pass(this) :: adam
  end type optimiser_type


  private

  public :: clip_type
  public :: optimiser_type


contains

!!!#############################################################################
!!! gradient norm clipping
!!!#############################################################################
  subroutine set_clip(this, clip_dict, clip_min, clip_max, clip_norm)
    implicit none
    class(optimiser_type), intent(inout) :: this
    type(clip_type), optional, intent(in) :: clip_dict
    real(real12), optional, intent(in) :: clip_min, clip_max, clip_norm


    !!--------------------------------------------------------------------------
    !! set up clipping limits
    !!--------------------------------------------------------------------------
    if(present(clip_dict))then
       this%clip_dict = clip_dict
       if(present(clip_min).or.present(clip_max).or.present(clip_norm))then
          write(*,*) "Multiple clip options provided to full layer"
          write(*,*) "Ignoring all bar clip_dict"
       end if
    else
       if(present(clip_min))then
          this%clip_dict%l_min_max = .true.
          this%clip_dict%min = clip_min
       end if
       if(present(clip_max))then
          this%clip_dict%l_min_max = .true.
          this%clip_dict%max = clip_max
       end if
       if(present(clip_norm))then
          this%clip_dict%l_norm = .true.
          this%clip_dict%norm = clip_norm
       end if
    end if

  end subroutine set_clip
!!!#############################################################################

!!!#############################################################################
!!! gradient norm clipping
!!!#############################################################################
  pure subroutine clip_gradients(this,length,gradient,bias)
    implicit none
    class(optimiser_type), intent(in) :: this
    integer, intent(in) :: length
    real(real12), dimension(length), intent(inout) :: gradient
    real(real12), dimension(:), optional, intent(inout) :: bias

    real(real12) :: scale
    real(real12), dimension(:), allocatable :: t_bias

    if(present(bias))then
       t_bias = bias
    else
       allocate(t_bias(1), source=0._real12)
    end if

    !! clip values to within limits of (min,max)
    if(this%clip_dict%l_min_max)then
       gradient = max(this%clip_dict%min,min(this%clip_dict%max,gradient))
       t_bias   = max(this%clip_dict%min,min(this%clip_dict%max,t_bias))
    end if

    !! clip values to a maximum L2-norm
    if(this%clip_dict%l_norm)then
       scale = min(1._real12, &
            this%clip_dict%norm/sqrt(sum(gradient**2._real12) + &
            sum(t_bias)**2._real12))
       if(scale.lt.1._real12)then
          gradient = gradient * scale
          t_bias   = t_bias * scale
       end if
    end if

    if(present(bias)) bias = t_bias

  end subroutine clip_gradients
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! 
!!!#############################################################################
  elemental subroutine optimise(this, weight, weight_incr, &
       gradient, m, v)
    implicit none
    class(optimiser_type), intent(in) :: this
    real(real12), intent(inout) :: weight
    real(real12), intent(inout) :: weight_incr
    real(real12), intent(inout) :: gradient
    real(real12), optional, intent(inout) :: m, v

    real(real12) :: t_learning_rate
    real(real12) :: lr_gradient

    lr_gradient = this%learning_rate * gradient

    !! adaptive learning method
    select case(this%method(1:1))
    case('m')!'momentum')
       !! momentum-based learning
       !! w = w - vel - lr * g
       weight_incr = lr_gradient + &
            this%momentum * weight_incr
    case('n')!('nesterov')
       !! nesterov momentum
       weight_incr = - this%momentum * weight_incr - &
            lr_gradient
    !case('a')!('adam')
    !   !! adam optimiser
    !   t_learning_rate = learning_rate
    !   call adam_optimiser(t_learning_rate, gradient, m, v)
    !   weight_incr = t_learning_rate
    case default
       weight_incr = lr_gradient
    end select

    !! regularisation
    if(this%regularise)then
       select case(this%regularisation)
       case('l1l2')
          !! L1L2 regularisation
          weight_incr = weight_incr + this%learning_rate * ( &
               this%l1 * sign(1._real12,weight) + &
               2._real12 * this%l2 * weight )
       case('l1')
          !! L1 regularisation
          weight_incr = weight_incr + this%learning_rate * &
               this%l1 * sign(1._real12,weight)
       case('l2')
          !! L2 regularisation
          weight_incr = weight_incr + this%learning_rate * &
               2._real12 * this%l2 * weight
       end select
    end if

    select case(this%method(1:1))
    case('n')!'nesterov')
       weight = weight + this%momentum * weight_incr - &
            lr_gradient
    case default
       weight = weight - weight_incr
    end select


  end subroutine optimise
!!!#############################################################################


!!!!############################################################################
!!!! adaptive learning rate
!!!! method: adam optimiser
!!!!         ... Adaptive Moment Estimation
!!!!############################################################################
!!!! learning_rate = initial learning rate hyperparameter
!!!! beta1 = exponential decay rate for first-moment estimates
!!!! beta2 = exponential decay rate for second-moment estimates
!!!! epsilon = small number for numerical stability
!!!! t = current iteration
!!!! m = first moment (m = mean of gradients)
!!!! v = second moment (v = variance of the gradients)
!  elemental subroutine adam(this, learning_rate, &
!       gradient, m, v)
!    implicit none
!    class(optimiser_type), intent(in) :: this
!    real(real12), intent(inout) :: learning_rate
!    real(real12), intent(in) :: gradient
!    real(real12), intent(inout) :: m, v
!
!    real :: m_norm, v_norm
!
!    !! update biased first moment estimate
!    m = this%beta1 * m + (1._real12 - this%beta1) * gradient
!
!    !! update biased second moment estimate
!    v = this%beta2 * v + (1._real12 - this%beta2) * gradient**2
!
!    !! normalised first moment estimate
!    m_norm = m / (1._real12 - this%beta1**this%iter)
!
!    !! normalised second moment estimate
!    v_norm = v / (1._real12 - this%beta2**this%iter)
!    
!    !! update learning rate
!    learning_rate = learning_rate * m_norm / (sqrt(v_norm) + this%epsilon)
!
!  end subroutine adam
!!!!############################################################################

  
  
  
end module optimiser
!!!#############################################################################

