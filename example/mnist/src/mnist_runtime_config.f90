module mnist_runtime_config
  use athena, only: base_optimiser_type, metric_dict_type, &
       sgd_optimiser_type, adam_optimiser_type, l1l2_regulariser_type
  use inputs, only: set_global_vars, metric_settings, optimiser_method, &
       optimiser_learning_rate, optimiser_momentum, optimiser_beta1, &
       optimiser_beta2, optimiser_epsilon, regularisation_method, &
       regularisation_l1, regularisation_l2, clip_min_value, &
       clip_max_value, clip_norm_value
  implicit none

  private

  public :: initialise_training_state

contains

  subroutine initialise_training_state(param_file, optimiser, metric_dict)
    character(*), intent(in) :: param_file
    class(base_optimiser_type), allocatable, intent(out) :: optimiser
    type(metric_dict_type), dimension(2), intent(out) :: metric_dict

    call set_global_vars(param_file=param_file)
    call build_metric_dict(metric_dict)
    call build_optimiser(optimiser)
  end subroutine initialise_training_state

  subroutine build_metric_dict(metric_dict)
    type(metric_dict_type), dimension(2), intent(out) :: metric_dict
    integer :: i

    do i=1,size(metric_dict)
       metric_dict(i)%active = metric_settings(i)%active
       metric_dict(i)%key = trim(metric_settings(i)%key)
       metric_dict(i)%threshold = metric_settings(i)%threshold
    end do
  end subroutine build_metric_dict

  subroutine build_optimiser(optimiser)
    class(base_optimiser_type), allocatable, intent(out) :: optimiser

    select case(trim(optimiser_method))
    case("", "none")
       allocate(optimiser, source=base_optimiser_type())
    case("sgd", "momentum")
       allocate(optimiser, source=sgd_optimiser_type( &
            momentum=optimiser_momentum))
    case("nesterov")
       allocate(optimiser, source=sgd_optimiser_type( &
            momentum=optimiser_momentum, nesterov=.true.))
    case("adam")
       allocate(optimiser, source=adam_optimiser_type( &
            beta1=optimiser_beta1, beta2=optimiser_beta2, &
            epsilon=optimiser_epsilon))
    case default
       stop "Unsupported optimiser method in MNIST example"
    end select

    call apply_regularisation(optimiser)
    optimiser%learning_rate = optimiser_learning_rate
    call optimiser%clip_dict%read( &
         clip_min_value, clip_max_value, clip_norm_value)
  end subroutine build_optimiser

  subroutine apply_regularisation(optimiser)
    class(base_optimiser_type), intent(inout) :: optimiser

    select case(trim(regularisation_method))
    case("", "none")
       optimiser%regularisation = .false.
    case("l1l2")
       optimiser%regularisation = .true.
       allocate(optimiser%regulariser, source=l1l2_regulariser_type( &
            l1=regularisation_l1, l2=regularisation_l2))
    case("l1")
       optimiser%regularisation = .true.
       allocate(optimiser%regulariser, source=l1l2_regulariser_type( &
            l1=regularisation_l1))
    case("l2")
       optimiser%regularisation = .true.
       allocate(optimiser%regulariser, source=l1l2_regulariser_type( &
            l2=regularisation_l2))
    case default
       stop "Unsupported regularisation method in MNIST example"
    end select
  end subroutine apply_regularisation

end module mnist_runtime_config
