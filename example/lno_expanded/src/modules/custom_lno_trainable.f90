module custom_lno_trainable
   use coreutils, only: real32
   use omp_lib, only: omp_get_max_threads, omp_get_thread_num, omp_in_parallel
  implicit none

  private

  public :: custom_lno_model_type
  public :: custom_lno_init
  public :: custom_lno_predict
  public :: custom_lno_train
  public :: custom_lno_rollout

  real(real32), parameter :: PI = 3.14159265358979323846_real32
  real(real32), parameter :: EPS_NORM = 1.0e-5_real32
  real(real32), parameter :: EPS_ADAM = 1.0e-8_real32

  type :: custom_lno_model_type
     integer :: grid_size = 112
     integer :: extended_grid = 114
     integer :: width = 64
     integer :: modes = 16
     integer :: num_blocks = 4
       integer :: num_input_channels = 1
     integer :: num_corrections = 3
     integer :: coeff_hidden = 32
     integer :: gate_hidden = 32
     real(real32) :: lr = 1.0e-3_real32
     real(real32) :: beta1 = 0.9_real32
     real(real32) :: beta2 = 0.999_real32
     real(real32) :: weight_decay = 1.0e-4_real32
     real(real32) :: grad_clip = 1.0_real32
       real(real32) :: filter_strength = 4.0_real32
     real(real32) :: max_amp = 1.0_real32
     real(real32) :: amp_sharpness = 1.0_real32
       real(real32) :: pole_min = 0.1_real32
       real(real32) :: pole_max = 100.0_real32
       real(real32) :: pole_offset_scale = 0.1_real32
       real(real32) :: causal_safety = 1.0_real32
       real(real32) :: max_correction_frac = 1.0e-2_real32
       real(real32) :: fo_scalar = 1.0_real32
       real(real32) :: tau_dt_scalar = 1.0_real32
       real(real32) :: ve_scalar = 1.0_real32
       real(real32) :: causal_max_dist_norm = huge(1.0_real32)
       real(real32) :: bc_left_norm = 0.0_real32
       real(real32) :: bc_right_norm = 0.0_real32
       logical :: use_causal_mask = .true.
       character(len=32) :: spectral_filter = 'exponential'

     real(real32), allocatable :: proj_w(:,:)
     real(real32), allocatable :: proj_b(:)
     real(real32), allocatable :: pw_w(:,:,:)
     real(real32), allocatable :: pw_b(:,:)
     real(real32), allocatable :: log_poles(:,:)
       real(real32), allocatable :: pole_mlp_w(:,:,:)
       real(real32), allocatable :: pole_mlp_b(:,:)
     real(real32), allocatable :: wt_log_amp(:,:,:,:)
     real(real32), allocatable :: wt_phase(:,:,:,:)
       real(real32), allocatable :: coeff1_w(:,:), coeff1_b(:)
       real(real32), allocatable :: coeff2_w(:,:), coeff2_b(:)
       real(real32), allocatable :: coeff3_w(:)
       real(real32) :: coeff3_b = 0.0_real32
      real(real32), allocatable :: coeff4_w(:)
      real(real32) :: coeff4_b = 0.0_real32
       real(real32), allocatable :: corr1_w(:,:), corr1_b(:)
       real(real32), allocatable :: corr2_w(:,:), corr2_b(:)
       real(real32), allocatable :: corr3_w(:)
       real(real32) :: corr3_b = 0.0_real32
       real(real32), allocatable :: step_sizes(:)
       real(real32), allocatable :: gate1_w(:,:), gate1_b(:)
       real(real32), allocatable :: gate2_w(:)
       real(real32) :: gate2_b = 0.0_real32
       real(real32) :: relax_log_strength = -2.94_real32
       real(real32), allocatable :: boundary_mask(:)
       real(real32), allocatable :: xi(:)
     real(real32), allocatable :: spec_filter(:)
     real(real32), allocatable :: dist(:,:)

     real(real32), allocatable :: m_proj_w(:,:), v_proj_w(:,:)
     real(real32), allocatable :: m_proj_b(:), v_proj_b(:)
     real(real32), allocatable :: m_pw_w(:,:,:), v_pw_w(:,:,:)
     real(real32), allocatable :: m_pw_b(:,:), v_pw_b(:,:)
     real(real32), allocatable :: m_log_poles(:,:), v_log_poles(:,:)
       real(real32), allocatable :: m_pole_mlp_w(:,:,:), v_pole_mlp_w(:,:,:)
       real(real32), allocatable :: m_pole_mlp_b(:,:), v_pole_mlp_b(:,:)
     real(real32), allocatable :: m_wt_log_amp(:,:,:,:), v_wt_log_amp(:,:,:,:)
     real(real32), allocatable :: m_wt_phase(:,:,:,:), v_wt_phase(:,:,:,:)
       real(real32), allocatable :: m_coeff1_w(:,:), v_coeff1_w(:,:)
       real(real32), allocatable :: m_coeff1_b(:), v_coeff1_b(:)
       real(real32), allocatable :: m_coeff2_w(:,:), v_coeff2_w(:,:)
       real(real32), allocatable :: m_coeff2_b(:), v_coeff2_b(:)
       real(real32), allocatable :: m_coeff3_w(:), v_coeff3_w(:)
       real(real32) :: m_coeff3_b = 0.0_real32
       real(real32) :: v_coeff3_b = 0.0_real32
      real(real32), allocatable :: m_coeff4_w(:), v_coeff4_w(:)
      real(real32) :: m_coeff4_b = 0.0_real32
      real(real32) :: v_coeff4_b = 0.0_real32
       real(real32), allocatable :: m_corr1_w(:,:), v_corr1_w(:,:)
       real(real32), allocatable :: m_corr1_b(:), v_corr1_b(:)
       real(real32), allocatable :: m_corr2_w(:,:), v_corr2_w(:,:)
       real(real32), allocatable :: m_corr2_b(:), v_corr2_b(:)
       real(real32), allocatable :: m_corr3_w(:), v_corr3_w(:)
       real(real32) :: m_corr3_b = 0.0_real32
       real(real32) :: v_corr3_b = 0.0_real32
       real(real32), allocatable :: m_step_sizes(:), v_step_sizes(:)
       real(real32), allocatable :: m_gate1_w(:,:), v_gate1_w(:,:)
       real(real32), allocatable :: m_gate1_b(:), v_gate1_b(:)
       real(real32), allocatable :: m_gate2_w(:), v_gate2_w(:)
       real(real32) :: m_gate2_b = 0.0_real32
       real(real32) :: v_gate2_b = 0.0_real32
       real(real32) :: m_relax_log_strength = 0.0_real32
       real(real32) :: v_relax_log_strength = 0.0_real32
     integer :: adam_t = 0
  end type custom_lno_model_type

  type :: block_cache_type
     real(real32), allocatable :: x_in(:,:)
     real(real32), allocatable :: x_hat(:,:)
     real(real32), allocatable :: mu(:)
     real(real32), allocatable :: inv_std(:)
     real(real32), allocatable :: pw_in(:,:)
     real(real32), allocatable :: pw_out(:,:)
     real(real32), allocatable :: act_out(:,:)
     real(real32), allocatable :: kernel_x(:,:,:)
     real(real32), allocatable :: kernels(:,:,:)
     real(real32), allocatable :: raw_kernels(:,:,:)
     real(real32), allocatable :: row_sum(:,:)
     real(real32), allocatable :: poles(:)
     real(real32), allocatable :: x_mean(:)
     real(real32), allocatable :: pole_offset(:)
     real(real32), allocatable :: amp(:,:,:)
     real(real32), allocatable :: weights(:,:,:)
  end type block_cache_type

  type :: grad_accum_type
     real(real32), allocatable :: proj_w(:,:), proj_b(:)
     real(real32), allocatable :: pw_w(:,:,:), pw_b(:,:)
     real(real32), allocatable :: log_poles(:,:), pole_mlp_w(:,:,:), pole_mlp_b(:,:)
     real(real32), allocatable :: wt_log_amp(:,:,:,:), wt_phase(:,:,:,:)
     real(real32), allocatable :: coeff1_w(:,:), coeff1_b(:)
     real(real32), allocatable :: coeff2_w(:,:), coeff2_b(:)
     real(real32), allocatable :: coeff3_w(:)
     real(real32) :: coeff3_b = 0.0_real32
   real(real32), allocatable :: coeff4_w(:)
   real(real32) :: coeff4_b = 0.0_real32
     real(real32), allocatable :: corr1_w(:,:), corr1_b(:)
     real(real32), allocatable :: corr2_w(:,:), corr2_b(:)
     real(real32), allocatable :: corr3_w(:)
     real(real32) :: corr3_b = 0.0_real32
     real(real32), allocatable :: step_sizes(:)
     real(real32), allocatable :: gate1_w(:,:), gate1_b(:)
     real(real32), allocatable :: gate2_w(:)
     real(real32) :: gate2_b = 0.0_real32
     real(real32) :: relax_log_strength = 0.0_real32
  end type grad_accum_type

contains

   subroutine allocate_gradients(model, grad)
      type(custom_lno_model_type), intent(in) :: model
      type(grad_accum_type), intent(inout) :: grad

         allocate(grad%proj_w(model%width, model%num_input_channels), grad%proj_b(model%width))
      allocate(grad%pw_w(model%width, model%width, model%num_blocks), grad%pw_b(model%width, model%num_blocks))
      allocate(grad%log_poles(model%modes, model%num_blocks))
      allocate(grad%pole_mlp_w(model%modes, model%width, model%num_blocks), grad%pole_mlp_b(model%modes, model%num_blocks))
      allocate(grad%wt_log_amp(model%modes, model%width, model%width, model%num_blocks))
      allocate(grad%wt_phase(model%modes, model%width, model%width, model%num_blocks))
        allocate(grad%coeff1_w(model%width, model%width + 2), grad%coeff1_b(model%width))
        allocate(grad%coeff2_w(model%coeff_hidden, model%width), grad%coeff2_b(model%coeff_hidden))
        allocate(grad%coeff3_w(model%coeff_hidden))
      allocate(grad%coeff4_w(model%coeff_hidden))
        allocate(grad%corr1_w(model%width, model%width + 1), grad%corr1_b(model%width))
        allocate(grad%corr2_w(model%coeff_hidden, model%width), grad%corr2_b(model%coeff_hidden))
        allocate(grad%corr3_w(model%coeff_hidden))
        allocate(grad%step_sizes(model%num_corrections))
        allocate(grad%gate1_w(model%gate_hidden, model%width + 1), grad%gate1_b(model%gate_hidden))
        allocate(grad%gate2_w(model%gate_hidden))
      call zero_gradients(grad)
   end subroutine allocate_gradients

   subroutine zero_gradients(grad)
      type(grad_accum_type), intent(inout) :: grad
      if (allocated(grad%proj_w)) grad%proj_w = 0.0_real32
      if (allocated(grad%proj_b)) grad%proj_b = 0.0_real32
      if (allocated(grad%pw_w)) grad%pw_w = 0.0_real32
      if (allocated(grad%pw_b)) grad%pw_b = 0.0_real32
      if (allocated(grad%log_poles)) grad%log_poles = 0.0_real32
      if (allocated(grad%pole_mlp_w)) grad%pole_mlp_w = 0.0_real32
      if (allocated(grad%pole_mlp_b)) grad%pole_mlp_b = 0.0_real32
      if (allocated(grad%wt_log_amp)) grad%wt_log_amp = 0.0_real32
      if (allocated(grad%wt_phase)) grad%wt_phase = 0.0_real32
        if (allocated(grad%coeff1_w)) grad%coeff1_w = 0.0_real32
        if (allocated(grad%coeff1_b)) grad%coeff1_b = 0.0_real32
        if (allocated(grad%coeff2_w)) grad%coeff2_w = 0.0_real32
        if (allocated(grad%coeff2_b)) grad%coeff2_b = 0.0_real32
        if (allocated(grad%coeff3_w)) grad%coeff3_w = 0.0_real32
      if (allocated(grad%coeff4_w)) grad%coeff4_w = 0.0_real32
        if (allocated(grad%corr1_w)) grad%corr1_w = 0.0_real32
        if (allocated(grad%corr1_b)) grad%corr1_b = 0.0_real32
        if (allocated(grad%corr2_w)) grad%corr2_w = 0.0_real32
        if (allocated(grad%corr2_b)) grad%corr2_b = 0.0_real32
        if (allocated(grad%corr3_w)) grad%corr3_w = 0.0_real32
        if (allocated(grad%step_sizes)) grad%step_sizes = 0.0_real32
        if (allocated(grad%gate1_w)) grad%gate1_w = 0.0_real32
        if (allocated(grad%gate1_b)) grad%gate1_b = 0.0_real32
        if (allocated(grad%gate2_w)) grad%gate2_w = 0.0_real32
        grad%coeff3_b = 0.0_real32
            grad%coeff4_b = 0.0_real32
        grad%corr3_b = 0.0_real32
        grad%gate2_b = 0.0_real32
      grad%relax_log_strength = 0.0_real32
   end subroutine zero_gradients

   subroutine add_gradients(dst, src)
      type(grad_accum_type), intent(inout) :: dst
      type(grad_accum_type), intent(in) :: src
      dst%proj_w = dst%proj_w + src%proj_w
      dst%proj_b = dst%proj_b + src%proj_b
      dst%pw_w = dst%pw_w + src%pw_w
      dst%pw_b = dst%pw_b + src%pw_b
      dst%log_poles = dst%log_poles + src%log_poles
      dst%pole_mlp_w = dst%pole_mlp_w + src%pole_mlp_w
      dst%pole_mlp_b = dst%pole_mlp_b + src%pole_mlp_b
      dst%wt_log_amp = dst%wt_log_amp + src%wt_log_amp
      dst%wt_phase = dst%wt_phase + src%wt_phase
        dst%coeff1_w = dst%coeff1_w + src%coeff1_w
        dst%coeff1_b = dst%coeff1_b + src%coeff1_b
        dst%coeff2_w = dst%coeff2_w + src%coeff2_w
        dst%coeff2_b = dst%coeff2_b + src%coeff2_b
        dst%coeff3_w = dst%coeff3_w + src%coeff3_w
        dst%coeff3_b = dst%coeff3_b + src%coeff3_b
      dst%coeff4_w = dst%coeff4_w + src%coeff4_w
      dst%coeff4_b = dst%coeff4_b + src%coeff4_b
        dst%corr1_w = dst%corr1_w + src%corr1_w
        dst%corr1_b = dst%corr1_b + src%corr1_b
        dst%corr2_w = dst%corr2_w + src%corr2_w
        dst%corr2_b = dst%corr2_b + src%corr2_b
        dst%corr3_w = dst%corr3_w + src%corr3_w
        dst%corr3_b = dst%corr3_b + src%corr3_b
        dst%step_sizes = dst%step_sizes + src%step_sizes
        dst%gate1_w = dst%gate1_w + src%gate1_w
        dst%gate1_b = dst%gate1_b + src%gate1_b
        dst%gate2_w = dst%gate2_w + src%gate2_w
        dst%gate2_b = dst%gate2_b + src%gate2_b
      dst%relax_log_strength = dst%relax_log_strength + src%relax_log_strength
   end subroutine add_gradients

   subroutine scale_gradients(grad, factor)
      type(grad_accum_type), intent(inout) :: grad
      real(real32), intent(in) :: factor
      grad%proj_w = grad%proj_w * factor
      grad%proj_b = grad%proj_b * factor
      grad%pw_w = grad%pw_w * factor
      grad%pw_b = grad%pw_b * factor
      grad%log_poles = grad%log_poles * factor
      grad%pole_mlp_w = grad%pole_mlp_w * factor
      grad%pole_mlp_b = grad%pole_mlp_b * factor
      grad%wt_log_amp = grad%wt_log_amp * factor
      grad%wt_phase = grad%wt_phase * factor
        grad%coeff1_w = grad%coeff1_w * factor
        grad%coeff1_b = grad%coeff1_b * factor
        grad%coeff2_w = grad%coeff2_w * factor
        grad%coeff2_b = grad%coeff2_b * factor
        grad%coeff3_w = grad%coeff3_w * factor
        grad%coeff3_b = grad%coeff3_b * factor
      grad%coeff4_w = grad%coeff4_w * factor
      grad%coeff4_b = grad%coeff4_b * factor
        grad%corr1_w = grad%corr1_w * factor
        grad%corr1_b = grad%corr1_b * factor
        grad%corr2_w = grad%corr2_w * factor
        grad%corr2_b = grad%corr2_b * factor
        grad%corr3_w = grad%corr3_w * factor
        grad%corr3_b = grad%corr3_b * factor
        grad%step_sizes = grad%step_sizes * factor
        grad%gate1_w = grad%gate1_w * factor
        grad%gate1_b = grad%gate1_b * factor
        grad%gate2_w = grad%gate2_w * factor
        grad%gate2_b = grad%gate2_b * factor
      grad%relax_log_strength = grad%relax_log_strength * factor
   end subroutine scale_gradients

   real(real32) function gradient_norm(grad)
      type(grad_accum_type), intent(in) :: grad
      gradient_norm = sqrt(sum(grad%proj_w ** 2) + sum(grad%proj_b ** 2) + &
             sum(grad%pw_w ** 2) + sum(grad%pw_b ** 2) + sum(grad%log_poles ** 2) + sum(grad%pole_mlp_w ** 2) + sum(grad%pole_mlp_b ** 2) + &
               sum(grad%wt_log_amp ** 2) + sum(grad%wt_phase ** 2) + sum(grad%coeff1_w ** 2) + sum(grad%coeff1_b ** 2) + &
               sum(grad%coeff2_w ** 2) + sum(grad%coeff2_b ** 2) + sum(grad%coeff3_w ** 2) + grad%coeff3_b ** 2 + &
               sum(grad%coeff4_w ** 2) + grad%coeff4_b ** 2 + &
               sum(grad%corr1_w ** 2) + sum(grad%corr1_b ** 2) + sum(grad%corr2_w ** 2) + sum(grad%corr2_b ** 2) + &
               sum(grad%corr3_w ** 2) + grad%corr3_b ** 2 + sum(grad%step_sizes ** 2) + sum(grad%gate1_w ** 2) + &
               sum(grad%gate1_b ** 2) + sum(grad%gate2_w ** 2) + grad%gate2_b ** 2 + grad%relax_log_strength ** 2)
   end function gradient_norm

   subroutine deallocate_gradients(grad)
      type(grad_accum_type), intent(inout) :: grad
      if (allocated(grad%proj_w)) deallocate(grad%proj_w)
      if (allocated(grad%proj_b)) deallocate(grad%proj_b)
      if (allocated(grad%pw_w)) deallocate(grad%pw_w)
      if (allocated(grad%pw_b)) deallocate(grad%pw_b)
      if (allocated(grad%log_poles)) deallocate(grad%log_poles)
      if (allocated(grad%pole_mlp_w)) deallocate(grad%pole_mlp_w)
      if (allocated(grad%pole_mlp_b)) deallocate(grad%pole_mlp_b)
      if (allocated(grad%wt_log_amp)) deallocate(grad%wt_log_amp)
      if (allocated(grad%wt_phase)) deallocate(grad%wt_phase)
        if (allocated(grad%coeff1_w)) deallocate(grad%coeff1_w)
        if (allocated(grad%coeff1_b)) deallocate(grad%coeff1_b)
        if (allocated(grad%coeff2_w)) deallocate(grad%coeff2_w)
        if (allocated(grad%coeff2_b)) deallocate(grad%coeff2_b)
        if (allocated(grad%coeff3_w)) deallocate(grad%coeff3_w)
      if (allocated(grad%coeff4_w)) deallocate(grad%coeff4_w)
        if (allocated(grad%corr1_w)) deallocate(grad%corr1_w)
        if (allocated(grad%corr1_b)) deallocate(grad%corr1_b)
        if (allocated(grad%corr2_w)) deallocate(grad%corr2_w)
        if (allocated(grad%corr2_b)) deallocate(grad%corr2_b)
        if (allocated(grad%corr3_w)) deallocate(grad%corr3_w)
        if (allocated(grad%step_sizes)) deallocate(grad%step_sizes)
        if (allocated(grad%gate1_w)) deallocate(grad%gate1_w)
        if (allocated(grad%gate1_b)) deallocate(grad%gate1_b)
        if (allocated(grad%gate2_w)) deallocate(grad%gate2_w)
   end subroutine deallocate_gradients

   subroutine apply_gradients(model, grad)
      type(custom_lno_model_type), intent(inout) :: model
      type(grad_accum_type), intent(in) :: grad
      real(real32) :: grad_norm, scale, bc1, bc2, gparam

      grad_norm = gradient_norm(grad)
      if (grad_norm > model%grad_clip) then
          scale = model%grad_clip / grad_norm
      else
          scale = 1.0_real32
      end if

      model%adam_t = model%adam_t + 1
      bc1 = 1.0_real32 - model%beta1 ** model%adam_t
      bc2 = 1.0_real32 - model%beta2 ** model%adam_t

      call adam_update_2d(model%proj_w, model%m_proj_w, model%v_proj_w, grad%proj_w * scale, model, bc1, bc2)
      call adam_update_1d(model%proj_b, model%m_proj_b, model%v_proj_b, grad%proj_b * scale, model, bc1, bc2)
      call adam_update_3d(model%pw_w, model%m_pw_w, model%v_pw_w, grad%pw_w * scale, model, bc1, bc2)
      call adam_update_2d(model%pw_b, model%m_pw_b, model%v_pw_b, grad%pw_b * scale, model, bc1, bc2)
      call adam_update_2d(model%log_poles, model%m_log_poles, model%v_log_poles, grad%log_poles * scale, model, bc1, bc2)
      call adam_update_3d(model%pole_mlp_w, model%m_pole_mlp_w, model%v_pole_mlp_w, grad%pole_mlp_w * scale, model, bc1, bc2)
      call adam_update_2d(model%pole_mlp_b, model%m_pole_mlp_b, model%v_pole_mlp_b, grad%pole_mlp_b * scale, model, bc1, bc2)
      call adam_update_4d(model%wt_log_amp, model%m_wt_log_amp, model%v_wt_log_amp, grad%wt_log_amp * scale, model, bc1, bc2)
      call adam_update_4d(model%wt_phase, model%m_wt_phase, model%v_wt_phase, grad%wt_phase * scale, model, bc1, bc2)
      call adam_update_2d(model%coeff1_w, model%m_coeff1_w, model%v_coeff1_w, grad%coeff1_w * scale, model, bc1, bc2)
      call adam_update_1d(model%coeff1_b, model%m_coeff1_b, model%v_coeff1_b, grad%coeff1_b * scale, model, bc1, bc2)
      call adam_update_2d(model%coeff2_w, model%m_coeff2_w, model%v_coeff2_w, grad%coeff2_w * scale, model, bc1, bc2)
      call adam_update_1d(model%coeff2_b, model%m_coeff2_b, model%v_coeff2_b, grad%coeff2_b * scale, model, bc1, bc2)
      call adam_update_1d(model%coeff3_w, model%m_coeff3_w, model%v_coeff3_w, grad%coeff3_w * scale, model, bc1, bc2)
      gparam = grad%coeff3_b * scale
      model%coeff3_b = model%coeff3_b * (1.0_real32 - model%lr * model%weight_decay)
      model%m_coeff3_b = model%beta1 * model%m_coeff3_b + (1.0_real32 - model%beta1) * gparam
      call adam_update_1d(model%coeff4_w, model%m_coeff4_w, model%v_coeff4_w, grad%coeff4_w * scale, model, bc1, bc2)
      gparam = grad%coeff4_b * scale
      model%coeff4_b = model%coeff4_b * (1.0_real32 - model%lr * model%weight_decay)
      model%m_coeff4_b = model%beta1 * model%m_coeff4_b + (1.0_real32 - model%beta1) * gparam
      model%v_coeff4_b = model%beta2 * model%v_coeff4_b + (1.0_real32 - model%beta2) * gparam * gparam
      model%coeff4_b = model%coeff4_b - model%lr * (model%m_coeff4_b / bc1) / (sqrt(model%v_coeff4_b / bc2) + EPS_ADAM)
      model%v_coeff3_b = model%beta2 * model%v_coeff3_b + (1.0_real32 - model%beta2) * gparam * gparam
      model%coeff3_b = model%coeff3_b - model%lr * (model%m_coeff3_b / bc1) / (sqrt(model%v_coeff3_b / bc2) + EPS_ADAM)
      call adam_update_2d(model%corr1_w, model%m_corr1_w, model%v_corr1_w, grad%corr1_w * scale, model, bc1, bc2)
      call adam_update_1d(model%corr1_b, model%m_corr1_b, model%v_corr1_b, grad%corr1_b * scale, model, bc1, bc2)
      call adam_update_2d(model%corr2_w, model%m_corr2_w, model%v_corr2_w, grad%corr2_w * scale, model, bc1, bc2)
      call adam_update_1d(model%corr2_b, model%m_corr2_b, model%v_corr2_b, grad%corr2_b * scale, model, bc1, bc2)
      call adam_update_1d(model%corr3_w, model%m_corr3_w, model%v_corr3_w, grad%corr3_w * scale, model, bc1, bc2)
      gparam = grad%corr3_b * scale
      model%corr3_b = model%corr3_b * (1.0_real32 - model%lr * model%weight_decay)
      model%m_corr3_b = model%beta1 * model%m_corr3_b + (1.0_real32 - model%beta1) * gparam
      model%v_corr3_b = model%beta2 * model%v_corr3_b + (1.0_real32 - model%beta2) * gparam * gparam
      model%corr3_b = model%corr3_b - model%lr * (model%m_corr3_b / bc1) / (sqrt(model%v_corr3_b / bc2) + EPS_ADAM)
      call adam_update_1d(model%step_sizes, model%m_step_sizes, model%v_step_sizes, grad%step_sizes * scale, model, bc1, bc2)
      call adam_update_2d(model%gate1_w, model%m_gate1_w, model%v_gate1_w, grad%gate1_w * scale, model, bc1, bc2)
      call adam_update_1d(model%gate1_b, model%m_gate1_b, model%v_gate1_b, grad%gate1_b * scale, model, bc1, bc2)
      call adam_update_1d(model%gate2_w, model%m_gate2_w, model%v_gate2_w, grad%gate2_w * scale, model, bc1, bc2)
      gparam = grad%gate2_b * scale
      model%gate2_b = model%gate2_b * (1.0_real32 - model%lr * model%weight_decay)
      model%m_gate2_b = model%beta1 * model%m_gate2_b + (1.0_real32 - model%beta1) * gparam
      model%v_gate2_b = model%beta2 * model%v_gate2_b + (1.0_real32 - model%beta2) * gparam * gparam
      model%gate2_b = model%gate2_b - model%lr * (model%m_gate2_b / bc1) / (sqrt(model%v_gate2_b / bc2) + EPS_ADAM)
      gparam = grad%relax_log_strength * scale
      model%m_relax_log_strength = model%beta1 * model%m_relax_log_strength + (1.0_real32 - model%beta1) * gparam
      model%v_relax_log_strength = model%beta2 * model%v_relax_log_strength + (1.0_real32 - model%beta2) * gparam * gparam
      model%relax_log_strength = model%relax_log_strength - model%lr * (model%m_relax_log_strength / bc1) / (sqrt(model%v_relax_log_strength / bc2) + EPS_ADAM)
   end subroutine apply_gradients

  elemental real(real32) function sigmoid(x)
    real(real32), intent(in) :: x
    if (x >= 0.0_real32) then
       sigmoid = 1.0_real32 / (1.0_real32 + exp(-x))
    else
       sigmoid = exp(x) / (1.0_real32 + exp(x))
    end if
  end function sigmoid

  elemental real(real32) function silu(x)
    real(real32), intent(in) :: x
    silu = x * sigmoid(x)
  end function silu

  elemental real(real32) function dsilu(x)
    real(real32), intent(in) :: x
    real(real32) :: s
    s = sigmoid(x)
    dsilu = s * (1.0_real32 + x * (1.0_real32 - s))
  end function dsilu

  elemental real(real32) function softplus(x)
    real(real32), intent(in) :: x
    if (x > 20.0_real32) then
       softplus = x
    else if (x < -20.0_real32) then
       softplus = exp(x)
    else
       softplus = log(1.0_real32 + exp(x))
    end if
  end function softplus

  subroutine random_normal_1d(arr, mean, std)
    real(real32), intent(out) :: arr(:)
    real(real32), intent(in) :: mean, std
    integer :: i
    real(real32) :: u1, u2
    do i = 1, size(arr), 2
       call random_number(u1)
       call random_number(u2)
       u1 = max(u1, 1.0e-10_real32)
       arr(i) = mean + std * sqrt(-2.0_real32 * log(u1)) * cos(2.0_real32 * PI * u2)
       if (i + 1 <= size(arr)) then
          arr(i + 1) = mean + std * sqrt(-2.0_real32 * log(u1)) * sin(2.0_real32 * PI * u2)
       end if
    end do
  end subroutine random_normal_1d

  subroutine random_normal_2d(arr, mean, std)
    real(real32), intent(out) :: arr(:,:)
    real(real32), intent(in) :: mean, std
    integer :: j
    do j = 1, size(arr, 2)
       call random_normal_1d(arr(:, j), mean, std)
    end do
  end subroutine random_normal_2d

  subroutine random_normal_3d(arr, mean, std)
    real(real32), intent(out) :: arr(:,:,:)
    real(real32), intent(in) :: mean, std
    integer :: k, j
    do k = 1, size(arr, 3)
       do j = 1, size(arr, 2)
          call random_normal_1d(arr(:, j, k), mean, std)
       end do
    end do
  end subroutine random_normal_3d

  subroutine random_normal_4d(arr, mean, std)
    real(real32), intent(out) :: arr(:,:,:,:)
    real(real32), intent(in) :: mean, std
    integer :: l, k, j
    do l = 1, size(arr, 4)
       do k = 1, size(arr, 3)
          do j = 1, size(arr, 2)
             call random_normal_1d(arr(:, j, k, l), mean, std)
          end do
       end do
    end do
  end subroutine random_normal_4d

  subroutine build_distance_matrix(dist)
    real(real32), intent(out) :: dist(:,:)
    integer :: i, j, g
    real(real32) :: denom
    g = size(dist, 1)
    denom = real(max(1, g - 1), real32)
    do j = 1, g
       do i = 1, g
          dist(i, j) = abs(real(i - 1, real32) - real(j - 1, real32)) / denom
       end do
    end do
  end subroutine build_distance_matrix

  subroutine build_filter(filter_values, spectral_filter, strength)
    real(real32), intent(out) :: filter_values(:)
    character(len=*), intent(in) :: spectral_filter
    real(real32), intent(in) :: strength
    integer :: k, m, cutoff
    real(real32) :: kn

    m = size(filter_values)
    if (m <= 1) then
       filter_values = 1.0_real32
       return
    end if

    select case (trim(spectral_filter))
    case ('none', 'None')
       filter_values = 1.0_real32
    case ('exponential')
       do k = 1, m
          kn = real(k - 1, real32) / real(m - 1, real32)
          filter_values(k) = exp(-strength * kn * kn)
       end do
    case ('raised_cosine')
       do k = 1, m
          kn = real(k - 1, real32) / real(m - 1, real32)
          filter_values(k) = 0.5_real32 * (1.0_real32 + cos(PI * kn))
       end do
    case ('sharp_cutoff')
       cutoff = int(real(m, real32) * 2.0_real32 / 3.0_real32)
       filter_values = 1.0_real32
       if (cutoff < m) filter_values(cutoff + 1:m) = 0.0_real32
    case ('dealias')
       cutoff = int(real(m, real32) * strength)
       cutoff = max(1, min(m, cutoff))
       filter_values = 1.0_real32
       if (cutoff < m) filter_values(cutoff + 1:m) = 0.0_real32
    case ('transient_optimized')
       cutoff = int(real(m, real32) * 0.8_real32)
       cutoff = max(1, min(m, cutoff))
       filter_values = 1.0_real32
       if (cutoff < m) then
          do k = cutoff + 1, m
             filter_values(k) = 1.0_real32 - real(k - cutoff, real32) / real(m - cutoff, real32)
          end do
       end if
    case default
       filter_values = 1.0_real32
    end select
  end subroutine build_filter

   subroutine extend_with_bc(interior, bc_left, bc_right, ext)
      real(real32), intent(in) :: interior(:)
      real(real32), intent(in) :: bc_left, bc_right
      real(real32), intent(out) :: ext(:)

      integer :: g

      g = size(interior)
      ext(1) = bc_left
      ext(2:g + 1) = interior
      ext(g + 2) = bc_right
   end subroutine extend_with_bc

   subroutine compute_sec_diff_extended(t_ext, sec_diff)
      real(real32), intent(in) :: t_ext(:)
      real(real32), intent(out) :: sec_diff(:)

      integer :: ge, i

      ge = size(t_ext)
      if (ge <= 1) then
          sec_diff = 0.0_real32
          return
      end if

      sec_diff(1) = t_ext(2) - t_ext(1)
      if (ge > 2) then
          do i = 2, ge - 1
               sec_diff(i) = t_ext(i + 1) - 2.0_real32 * t_ext(i) + t_ext(i - 1)
          end do
      end if
      sec_diff(ge) = t_ext(ge - 1) - t_ext(ge)
   end subroutine compute_sec_diff_extended

   subroutine set_runtime_conditions(model, bc_left_norm, bc_right_norm, alpha_scalar, tau_scalar, dt, dx)
      type(custom_lno_model_type), intent(inout) :: model
      real(real32), intent(in) :: bc_left_norm, bc_right_norm, alpha_scalar, tau_scalar, dt, dx
      real(real32) :: domain_length

      model%bc_left_norm = bc_left_norm
      model%bc_right_norm = bc_right_norm
      domain_length = max(dx * real(model%grid_size, real32), 1.0e-20_real32)
      model%fo_scalar = min(500.0_real32, max(1.0e-6_real32, alpha_scalar * dt / max(dx * dx, 1.0e-20_real32)))
      model%tau_dt_scalar = tau_scalar / max(dt, 1.0e-20_real32)
      model%ve_scalar = sqrt(max(alpha_scalar * tau_scalar, 0.0_real32)) / domain_length
      if (model%use_causal_mask) then
         model%causal_max_dist_norm = model%causal_safety * dt * sqrt(max(alpha_scalar / max(tau_scalar, 1.0e-30_real32), 0.0_real32)) / domain_length
      else
         model%causal_max_dist_norm = huge(1.0_real32)
      end if
   end subroutine set_runtime_conditions

   real(real32) function cosine_annealed_lr(initial_lr, epoch_index, num_epochs, min_ratio)
      real(real32), intent(in) :: initial_lr, min_ratio
      integer, intent(in) :: epoch_index, num_epochs
      real(real32) :: eta_min, phase

      eta_min = initial_lr * min_ratio
      phase = PI * real(max(0, epoch_index), real32) / real(max(1, num_epochs), real32)
      cosine_annealed_lr = eta_min + 0.5_real32 * (initial_lr - eta_min) * (1.0_real32 + cos(phase))
   end function cosine_annealed_lr

   subroutine build_steady_state_input(model, bc_left_norm, bc_right_norm, input_vec)
      type(custom_lno_model_type), intent(in) :: model
      real(real32), intent(in) :: bc_left_norm, bc_right_norm
      real(real32), intent(out) :: input_vec(:)

      integer :: g, i
      real(real32) :: steady_value

      g = model%grid_size
      do i = 1, g
         steady_value = bc_left_norm + (bc_right_norm - bc_left_norm) * model%xi(i)
         input_vec(i) = steady_value
         input_vec(g + i) = steady_value
      end do
   end subroutine build_steady_state_input

    subroutine custom_lno_init(model, grid_size, width, modes, num_blocks, lr, bc_left_norm, bc_right_norm, alpha, tau, dt, dx, &
            num_corrections, use_causal_mask, spectral_filter, filter_strength)
    type(custom_lno_model_type), intent(inout) :: model
    integer, intent(in) :: grid_size, width, modes, num_blocks
    integer, intent(in), optional :: num_corrections
    real(real32), intent(in), optional :: lr
      real(real32), intent(in), optional :: bc_left_norm, bc_right_norm, alpha, tau, dt, dx, filter_strength
      logical, intent(in), optional :: use_causal_mask
      character(len=*), intent(in), optional :: spectral_filter
    integer :: b, k
      real(real32) :: init_scale, target_amp, target_sigmoid, init_bias, xi

    model%grid_size = grid_size
      model%extended_grid = grid_size + 2
    model%width = width
    model%modes = modes
    model%num_blocks = num_blocks
         if (present(num_corrections)) model%num_corrections = max(1, num_corrections)
      model%coeff_hidden = max(1, width / 2)
      model%gate_hidden = max(1, width / 2)
    if (present(lr)) model%lr = lr
      if (present(bc_left_norm)) model%bc_left_norm = bc_left_norm
      if (present(bc_right_norm)) model%bc_right_norm = bc_right_norm
      if (present(filter_strength)) model%filter_strength = filter_strength
      if (present(use_causal_mask)) model%use_causal_mask = use_causal_mask
      if (present(spectral_filter)) model%spectral_filter = trim(spectral_filter)

      allocate(model%proj_w(width, model%num_input_channels), model%proj_b(width))
    allocate(model%pw_w(width, width, num_blocks), model%pw_b(width, num_blocks))
    allocate(model%log_poles(modes, num_blocks))
      allocate(model%pole_mlp_w(modes, width, num_blocks), model%pole_mlp_b(modes, num_blocks))
    allocate(model%wt_log_amp(modes, width, width, num_blocks))
    allocate(model%wt_phase(modes, width, width, num_blocks))
      allocate(model%coeff1_w(width, width + 2), model%coeff1_b(width))
      allocate(model%coeff2_w(model%coeff_hidden, width), model%coeff2_b(model%coeff_hidden))
      allocate(model%coeff3_w(model%coeff_hidden))
      allocate(model%coeff4_w(model%coeff_hidden))
      allocate(model%corr1_w(width, width + 1), model%corr1_b(width))
      allocate(model%corr2_w(model%coeff_hidden, width), model%corr2_b(model%coeff_hidden))
      allocate(model%corr3_w(model%coeff_hidden))
      allocate(model%step_sizes(model%num_corrections))
      allocate(model%gate1_w(model%gate_hidden, width + 1), model%gate1_b(model%gate_hidden))
      allocate(model%gate2_w(model%gate_hidden))
      allocate(model%boundary_mask(grid_size), model%xi(grid_size), model%spec_filter(modes), model%dist(model%extended_grid, model%extended_grid))

      allocate(model%m_proj_w(width, model%num_input_channels), model%v_proj_w(width, model%num_input_channels), source=0.0_real32)
    allocate(model%m_proj_b(width), model%v_proj_b(width), source=0.0_real32)
    allocate(model%m_pw_w(width, width, num_blocks), model%v_pw_w(width, width, num_blocks), source=0.0_real32)
    allocate(model%m_pw_b(width, num_blocks), model%v_pw_b(width, num_blocks), source=0.0_real32)
    allocate(model%m_log_poles(modes, num_blocks), model%v_log_poles(modes, num_blocks), source=0.0_real32)
    allocate(model%m_pole_mlp_w(modes, width, num_blocks), model%v_pole_mlp_w(modes, width, num_blocks), source=0.0_real32)
    allocate(model%m_pole_mlp_b(modes, num_blocks), model%v_pole_mlp_b(modes, num_blocks), source=0.0_real32)
    allocate(model%m_wt_log_amp(modes, width, width, num_blocks), model%v_wt_log_amp(modes, width, width, num_blocks), source=0.0_real32)
    allocate(model%m_wt_phase(modes, width, width, num_blocks), model%v_wt_phase(modes, width, width, num_blocks), source=0.0_real32)
      allocate(model%m_coeff1_w(width, width + 2), model%v_coeff1_w(width, width + 2), source=0.0_real32)
      allocate(model%m_coeff1_b(width), model%v_coeff1_b(width), source=0.0_real32)
      allocate(model%m_coeff2_w(model%coeff_hidden, width), model%v_coeff2_w(model%coeff_hidden, width), source=0.0_real32)
      allocate(model%m_coeff2_b(model%coeff_hidden), model%v_coeff2_b(model%coeff_hidden), source=0.0_real32)
      allocate(model%m_coeff3_w(model%coeff_hidden), model%v_coeff3_w(model%coeff_hidden), source=0.0_real32)
      allocate(model%m_coeff4_w(model%coeff_hidden), model%v_coeff4_w(model%coeff_hidden), source=0.0_real32)
      allocate(model%m_corr1_w(width, width + 1), model%v_corr1_w(width, width + 1), source=0.0_real32)
      allocate(model%m_corr1_b(width), model%v_corr1_b(width), source=0.0_real32)
      allocate(model%m_corr2_w(model%coeff_hidden, width), model%v_corr2_w(model%coeff_hidden, width), source=0.0_real32)
      allocate(model%m_corr2_b(model%coeff_hidden), model%v_corr2_b(model%coeff_hidden), source=0.0_real32)
      allocate(model%m_corr3_w(model%coeff_hidden), model%v_corr3_w(model%coeff_hidden), source=0.0_real32)
      allocate(model%m_step_sizes(model%num_corrections), model%v_step_sizes(model%num_corrections), source=0.0_real32)
      allocate(model%m_gate1_w(model%gate_hidden, width + 1), model%v_gate1_w(model%gate_hidden, width + 1), source=0.0_real32)
      allocate(model%m_gate1_b(model%gate_hidden), model%v_gate1_b(model%gate_hidden), source=0.0_real32)
      allocate(model%m_gate2_w(model%gate_hidden), model%v_gate2_w(model%gate_hidden), source=0.0_real32)

    model%causal_max_dist_norm = huge(1.0_real32)
    if (present(alpha) .and. present(tau) .and. present(dt) .and. present(dx)) then
       call set_runtime_conditions(model, model%bc_left_norm, model%bc_right_norm, alpha, tau, dt, dx)
    end if

    call random_normal_2d(model%proj_w, 0.0_real32, sqrt(2.0_real32 / real(model%num_input_channels, real32)))
    model%proj_b = 0.0_real32

    target_amp = sqrt(2.0_real32 / real(max(1, modes * width), real32))
    target_sigmoid = min(0.99_real32, max(0.01_real32, target_amp / model%max_amp))
    init_bias = log(target_sigmoid / (1.0_real32 - target_sigmoid)) / model%amp_sharpness
    init_scale = 1.0_real32 / sqrt(real(max(1, width * width * modes), real32))

    do b = 1, num_blocks
       call random_normal_2d(model%pw_w(:, :, b), 0.0_real32, sqrt(2.0_real32 / real(width, real32)))
       model%pw_b(:, b) = 0.0_real32
       call random_normal_2d(model%pole_mlp_w(:, :, b), 0.0_real32, 1.0_real32 / sqrt(real(width, real32)))
       model%pole_mlp_b(:, b) = 0.0_real32
       do k = 1, modes
          model%log_poles(k, b) = log(1.0_real32) + &
               (log(50.0_real32) - log(1.0_real32)) * real(k - 1, real32) / real(max(1, modes - 1), real32)
       end do
       call random_normal_3d(model%wt_log_amp(:, :, :, b), init_bias, init_scale)
       call random_normal_3d(model%wt_phase(:, :, :, b), 0.0_real32, 1.0_real32)
    end do

    call random_normal_2d(model%coeff1_w, 0.0_real32, sqrt(2.0_real32 / real(width + 2, real32)))
    model%coeff1_b = 0.0_real32
    call random_normal_2d(model%coeff2_w, 0.0_real32, sqrt(2.0_real32 / real(width, real32)))
    model%coeff2_b = 0.0_real32
    call random_normal_1d(model%coeff3_w, 0.0_real32, 1.0e-3_real32)
    model%coeff3_b = 0.0_real32
   call random_normal_1d(model%coeff4_w, 0.0_real32, 1.0e-3_real32)
   model%coeff4_b = 0.0_real32
    call random_normal_2d(model%corr1_w, 0.0_real32, sqrt(2.0_real32 / real(width + 1, real32)))
    model%corr1_b = 0.0_real32
    call random_normal_2d(model%corr2_w, 0.0_real32, sqrt(2.0_real32 / real(width, real32)))
    model%corr2_b = 0.0_real32
    call random_normal_1d(model%corr3_w, 0.0_real32, 1.0e-3_real32)
    model%corr3_b = 0.0_real32
    model%step_sizes = 0.1_real32
    call random_normal_2d(model%gate1_w, 0.0_real32, sqrt(2.0_real32 / real(width + 1, real32)))
    model%gate1_b = 0.0_real32
    call random_normal_1d(model%gate2_w, 0.0_real32, 1.0e-3_real32)
    model%gate2_b = 0.0_real32
    model%relax_log_strength = -2.94_real32
    model%boundary_mask = 1.0_real32
    model%boundary_mask(1) = 0.0_real32
    model%boundary_mask(grid_size) = 0.0_real32
    do k = 1, grid_size
       xi = real(k - 1, real32) / real(max(1, grid_size - 1), real32)
       model%xi(k) = xi
    end do
    call build_filter(model%spec_filter, model%spectral_filter, model%filter_strength)
    call build_distance_matrix(model%dist)
    model%adam_t = 0
    model%m_coeff3_b = 0.0_real32
    model%v_coeff3_b = 0.0_real32
   model%m_coeff4_b = 0.0_real32
   model%v_coeff4_b = 0.0_real32
    model%m_corr3_b = 0.0_real32
    model%v_corr3_b = 0.0_real32
    model%m_gate2_b = 0.0_real32
    model%v_gate2_b = 0.0_real32
    model%m_relax_log_strength = 0.0_real32
    model%v_relax_log_strength = 0.0_real32
  end subroutine custom_lno_init

   subroutine forward_with_cache(model, input, output, caches, x0, x_final)
    type(custom_lno_model_type), intent(in) :: model
    real(real32), intent(in) :: input(:)
    real(real32), intent(out) :: output(:)
    type(block_cache_type), allocatable, intent(out) :: caches(:)
    real(real32), allocatable, intent(out) :: x0(:,:)
      real(real32), allocatable, intent(out) :: x_final(:,:)

    integer :: g, ge, w, b, c, i, step_idx
    real(real32), allocatable :: x(:,:), t_n(:), t_nm1(:), t_n_ext(:), t_nm1_ext(:)
   real(real32), allocatable :: sec_diff_ext(:), diff_ext(:), a_theta_ext(:), b_theta_ext(:)
    real(real32), allocatable :: t_inc_init_ext(:), t_inc_ext(:), signal_gate(:)
   real(real32), allocatable :: coeff_in_mat(:,:), coeff_h1(:,:), coeff_h2(:,:)
   real(real32), allocatable :: corr_in_mat(:,:), corr_h1(:,:), corr_h2(:,:)
   real(real32), allocatable :: gate_in_mat(:,:), gate_h1(:,:)
    real(real32), allocatable :: gated_inc(:), relax_dir(:), gate(:), g_inc(:)
   real(real32) :: a_target, b_target, bound_a, bound_b, relax_strength, gate_scale, fo_log

    g = model%grid_size
    ge = model%extended_grid
    w = model%width
    allocate(caches(model%num_blocks))
    allocate(t_n(g), t_nm1(g), t_n_ext(ge), t_nm1_ext(ge))
    allocate(x(w, ge), x0(w, ge), x_final(w, ge))
   allocate(sec_diff_ext(ge), diff_ext(ge), a_theta_ext(ge), b_theta_ext(ge))
    allocate(t_inc_init_ext(ge), t_inc_ext(ge), signal_gate(ge))
   allocate(coeff_in_mat(w + 2, ge), coeff_h1(w, ge), coeff_h2(model%coeff_hidden, ge))
   allocate(corr_in_mat(w + 1, ge), corr_h1(w, ge), corr_h2(model%coeff_hidden, ge))
   allocate(gate_in_mat(w + 1, g), gate_h1(model%gate_hidden, g))
    allocate(gated_inc(g), relax_dir(g), gate(g), g_inc(g))
    t_n = input(1:g)
    t_nm1 = input(g + 1:2 * g)

    call extend_with_bc(t_n, model%bc_left_norm, model%bc_right_norm, t_n_ext)
    call extend_with_bc(t_nm1, model%bc_left_norm, model%bc_right_norm, t_nm1_ext)
    diff_ext = t_n_ext - t_nm1_ext
    call compute_sec_diff_extended(t_n_ext, sec_diff_ext)
    fo_log = log(1.0_real32 + model%fo_scalar)

   !$omp parallel do collapse(2) schedule(static) if(.not. omp_in_parallel()) default(shared) private(c,i)
    do c = 1, w
       do i = 1, ge
          x(c, i) = model%proj_w(c, 1) * t_n_ext(i) + model%proj_b(c)
       end do
    end do
    !$omp end parallel do
    x0 = x

    do b = 1, model%num_blocks
       call lno_block_forward(model, b, x, caches(b))
    end do

    x_final = x

    a_target = model%fo_scalar / (1.0_real32 + model%tau_dt_scalar)
    b_target = model%tau_dt_scalar / (1.0_real32 + model%tau_dt_scalar)
    bound_a = model%max_correction_frac * max(abs(a_target), 1.0e-6_real32)
   bound_b = model%max_correction_frac * max(abs(b_target), 1.0e-6_real32)

    coeff_in_mat(1:w, :) = x
    coeff_in_mat(w + 1, :) = fo_log
    coeff_in_mat(w + 2, :) = model%ve_scalar
    coeff_h1 = silu(matmul(model%coeff1_w, coeff_in_mat) + spread(model%coeff1_b, dim=2, ncopies=ge))
    coeff_h2 = silu(matmul(model%coeff2_w, coeff_h1) + spread(model%coeff2_b, dim=2, ncopies=ge))
    do i = 1, ge
       a_theta_ext(i) = a_target + bound_a * tanh(dot_product(model%coeff3_w, coeff_h2(:, i)) + model%coeff3_b)
       b_theta_ext(i) = b_target + bound_b * tanh(dot_product(model%coeff4_w, coeff_h2(:, i)) + model%coeff4_b)
       t_inc_init_ext(i) = a_theta_ext(i) * sec_diff_ext(i) + b_theta_ext(i) * diff_ext(i)
    end do

    t_inc_ext = t_inc_init_ext
    gate_scale = max(maxval(abs(t_inc_init_ext)), 1.0e-12_real32)
    signal_gate = abs(t_inc_init_ext) / gate_scale
    do step_idx = 1, model%num_corrections
       corr_in_mat(1:w, :) = x
       corr_in_mat(w + 1, :) = t_inc_ext
       corr_h1 = silu(matmul(model%corr1_w, corr_in_mat) + spread(model%corr1_b, dim=2, ncopies=ge))
       corr_h2 = silu(matmul(model%corr2_w, corr_h1) + spread(model%corr2_b, dim=2, ncopies=ge))
       do i = 1, ge
          t_inc_ext(i) = t_inc_ext(i) + model%step_sizes(step_idx) * (dot_product(model%corr3_w, corr_h2(:, i)) + model%corr3_b) * signal_gate(i)
       end do
    end do

    relax_strength = softplus(model%relax_log_strength)
    gate_in_mat(1:w, :) = x_final(:, 2:g + 1)
    gate_in_mat(w + 1, :) = t_inc_ext(2:g + 1)
    gate_h1 = silu(matmul(model%gate1_w, gate_in_mat) + spread(model%gate1_b, dim=2, ncopies=g))
    do i = 1, g
       relax_dir(i) = model%bc_left_norm + (model%bc_right_norm - model%bc_left_norm) * model%xi(i) - t_n(i)
       gate(i) = sigmoid(dot_product(model%gate2_w, gate_h1(:, i)) + model%gate2_b)
       gated_inc(i) = t_inc_ext(i + 1) + gate(i) * relax_strength * relax_dir(i)
       g_inc(i) = (model%bc_left_norm - t_n(1)) + ((model%bc_right_norm - t_n(g)) - (model%bc_left_norm - t_n(1))) * model%xi(i)
       output(i) = g_inc(i) + model%boundary_mask(i) * gated_inc(i)
    end do

   deallocate(t_n, t_nm1, t_n_ext, t_nm1_ext, x, sec_diff_ext, diff_ext, a_theta_ext, b_theta_ext, t_inc_init_ext, t_inc_ext, signal_gate, &
         coeff_in_mat, coeff_h1, coeff_h2, corr_in_mat, corr_h1, corr_h2, gate_in_mat, gate_h1, gated_inc, relax_dir, gate, g_inc)
  end subroutine forward_with_cache

  subroutine lno_block_forward(model, block_idx, x, cache)
    type(custom_lno_model_type), intent(in) :: model
    integer, intent(in) :: block_idx
    real(real32), intent(inout) :: x(:,:)
    type(block_cache_type), intent(inout) :: cache

    integer :: g, w, m, c, i, j, k, co, ci
   real(real32) :: mu, var, inv_std, pole, row_sum, amp, pole_preact
    real(real32), allocatable :: out(:,:)

   g = model%extended_grid
    w = model%width
    m = model%modes

    allocate(cache%x_in(w, g), cache%x_hat(w, g), cache%mu(w), cache%inv_std(w))
    allocate(cache%pw_in(w, g), cache%pw_out(w, g), cache%act_out(w, g))
    allocate(cache%kernel_x(m, w, g), cache%kernels(m, g, g), cache%raw_kernels(m, g, g))
   allocate(cache%row_sum(m, g), cache%poles(m), cache%x_mean(w), cache%pole_offset(m), cache%amp(m, w, w), cache%weights(m, w, w))
    allocate(out(w, g))

    cache%x_in = x
   !$omp parallel do schedule(static) if(.not. omp_in_parallel()) default(shared) private(c,mu,var,inv_std)
   do c = 1, w
       mu = sum(x(c, :)) / real(g, real32)
       var = sum((x(c, :) - mu) ** 2) / real(g, real32)
       inv_std = 1.0_real32 / sqrt(var + EPS_NORM)
       cache%mu(c) = mu
       cache%inv_std(c) = inv_std
       cache%x_hat(c, :) = (x(c, :) - mu) * inv_std
         x(c, :) = cache%x_hat(c, :)
    end do
    !$omp end parallel do

    cache%pw_in = x
    cache%pw_out = matmul(model%pw_w(:, :, block_idx), x) + spread(model%pw_b(:, block_idx), dim=2, ncopies=g)
    cache%act_out = silu(cache%pw_out)
   cache%x_mean = sum(cache%act_out, dim=2) / real(g, real32)

   !$omp parallel do schedule(static) if(.not. omp_in_parallel()) default(shared) private(k,pole,i,j,row_sum,co,ci,amp,pole_preact)
   do k = 1, m
      cache%pole_offset(k) = tanh(sum(model%pole_mlp_w(k, :, block_idx) * cache%x_mean) + model%pole_mlp_b(k, block_idx))
      pole_preact = model%log_poles(k, block_idx) + model%pole_offset_scale * cache%pole_offset(k)
      pole = min(model%pole_max, max(model%pole_min, softplus(pole_preact)))
       cache%poles(k) = pole
       do i = 1, g
          row_sum = 0.0_real32
          do j = 1, g
             if (model%dist(i, j) <= model%causal_max_dist_norm) then
                cache%raw_kernels(k, i, j) = exp(-pole * model%dist(i, j))
             else
                cache%raw_kernels(k, i, j) = 0.0_real32
             end if
             row_sum = row_sum + cache%raw_kernels(k, i, j)
          end do
          cache%row_sum(k, i) = max(row_sum, 1.0e-8_real32)
          cache%kernels(k, i, :) = (cache%raw_kernels(k, i, :) / cache%row_sum(k, i)) * model%spec_filter(k)
       end do
       do co = 1, w
          do ci = 1, w
             amp = model%max_amp * sigmoid(model%amp_sharpness * model%wt_log_amp(k, ci, co, block_idx))
             cache%amp(k, ci, co) = amp
             cache%weights(k, ci, co) = amp * cos(model%wt_phase(k, ci, co, block_idx))
          end do
       end do
    end do
    !$omp end parallel do

    do k = 1, m
       cache%kernel_x(k, :, :) = matmul(cache%act_out, transpose(cache%kernels(k, :, :)))
    end do

    out = 0.0_real32
    do k = 1, m
       out = out + matmul(transpose(cache%weights(k, :, :)), cache%kernel_x(k, :, :))
    end do
    x = out
    deallocate(out)
  end subroutine lno_block_forward

  subroutine custom_lno_predict(model, input, output)
    type(custom_lno_model_type), intent(in) :: model
    real(real32), intent(in) :: input(:)
    real(real32), intent(out) :: output(:)
    type(block_cache_type), allocatable :: caches(:)
      real(real32), allocatable :: x0(:,:), x_final(:,:)
      call forward_with_cache(model, input, output, caches, x0, x_final)
      call free_caches(caches, x0, x_final)
  end subroutine custom_lno_predict

   subroutine compute_gradients(model, input, target, grad, sample_loss)
      type(custom_lno_model_type), intent(in) :: model
      real(real32), intent(in) :: input(:), target(:)
      type(grad_accum_type), intent(inout) :: grad
      real(real32), intent(out) :: sample_loss

      type(block_cache_type), allocatable :: caches(:)
      real(real32), allocatable :: x0(:,:), x_final(:,:), pred(:), dx(:,:), dprev(:,:)
      real(real32), allocatable :: t_n(:), t_nm1(:), t_n_ext(:), t_nm1_ext(:)
      real(real32), allocatable :: sec_diff_ext(:), diff_ext(:), signal_gate(:), dy(:)
      real(real32), allocatable :: coeff1_pre(:,:), coeff1_act(:,:), coeff2_pre(:,:), coeff2_act(:,:), raw_a(:), raw_b(:)
      real(real32), allocatable :: t_iter(:,:), corr_delta(:,:)
      real(real32), allocatable :: corr1_pre(:,:,:), corr1_act(:,:,:), corr2_pre(:,:,:), corr2_act(:,:,:)
      real(real32), allocatable :: gate1_pre(:,:), gate1_act(:,:), gate(:), relax_dir(:)
      real(real32), allocatable :: coeff_in(:), corr_in(:), gate_in(:)
      real(real32), allocatable :: d_h1(:), d_h1_pre(:), d_h2(:), d_h2_pre(:)
      real(real32), allocatable :: d_gate_h1(:), d_gate_h1_pre(:), xmean_grad(:)
      real(real32), allocatable :: dkernel_x(:,:,:), dkernels(:,:,:), draw(:,:,:), dact(:,:), dz(:,:), dnorm(:,:)
      real(real32) :: loss_scale, a_target, b_target, bound_a, bound_b, relax_strength, fo_log
      real(real32) :: d_gated, d_gate_pre, d_tinc, raw_a_grad, raw_b_grad
      real(real32) :: centered_sum, xhat_sum, sig, dpole, pole_preact, pole_offset_grad
      integer :: g, ge, w, m, gh, ch, i, b, k, c, ci, co, step_idx, idx

      g = model%grid_size
      ge = model%extended_grid
      w = model%width
      m = model%modes
      gh = model%gate_hidden
      ch = model%coeff_hidden

      call zero_gradients(grad)
      allocate(pred(g), dx(w, ge), dprev(w, ge))
      allocate(t_n(g), t_nm1(g), t_n_ext(ge), t_nm1_ext(ge), sec_diff_ext(ge), diff_ext(ge), signal_gate(ge), dy(g))
      allocate(coeff1_pre(w, ge), coeff1_act(w, ge), coeff2_pre(ch, ge), coeff2_act(ch, ge), raw_a(ge), raw_b(ge))
      allocate(t_iter(0:model%num_corrections, ge), corr_delta(model%num_corrections, ge))
      allocate(corr1_pre(w, model%num_corrections, ge), corr1_act(w, model%num_corrections, ge))
      allocate(corr2_pre(ch, model%num_corrections, ge), corr2_act(ch, model%num_corrections, ge))
      allocate(gate1_pre(gh, g), gate1_act(gh, g), gate(g), relax_dir(g))
      allocate(coeff_in(w + 2), corr_in(w + 1), gate_in(w + 1))
      allocate(d_h1(w), d_h1_pre(w), d_h2(ch), d_h2_pre(ch), d_gate_h1(gh), d_gate_h1_pre(gh), xmean_grad(w))

      call forward_with_cache(model, input, pred, caches, x0, x_final)
      sample_loss = sum((pred - target) ** 2) / real(g, real32)

      t_n = input(1:g)
      t_nm1 = input(g + 1:2 * g)
      call extend_with_bc(t_n, model%bc_left_norm, model%bc_right_norm, t_n_ext)
      call extend_with_bc(t_nm1, model%bc_left_norm, model%bc_right_norm, t_nm1_ext)
      diff_ext = t_n_ext - t_nm1_ext
      call compute_sec_diff_extended(t_n_ext, sec_diff_ext)

      fo_log = log(1.0_real32 + model%fo_scalar)
      a_target = model%fo_scalar / (1.0_real32 + model%tau_dt_scalar)
      b_target = model%tau_dt_scalar / (1.0_real32 + model%tau_dt_scalar)
      bound_a = model%max_correction_frac * max(abs(a_target), 1.0e-6_real32)
      bound_b = model%max_correction_frac * max(abs(b_target), 1.0e-6_real32)

      do i = 1, ge
         coeff_in(1:w) = x_final(:, i)
         coeff_in(w + 1) = fo_log
         coeff_in(w + 2) = model%ve_scalar
         coeff1_pre(:, i) = matmul(model%coeff1_w, coeff_in) + model%coeff1_b
         coeff1_act(:, i) = silu(coeff1_pre(:, i))
         coeff2_pre(:, i) = matmul(model%coeff2_w, coeff1_act(:, i)) + model%coeff2_b
         coeff2_act(:, i) = silu(coeff2_pre(:, i))
         raw_a(i) = dot_product(model%coeff3_w, coeff2_act(:, i)) + model%coeff3_b
         raw_b(i) = dot_product(model%coeff4_w, coeff2_act(:, i)) + model%coeff4_b
         t_iter(0, i) = (a_target + bound_a * tanh(raw_a(i))) * sec_diff_ext(i) + (b_target + bound_b * tanh(raw_b(i))) * diff_ext(i)
      end do

      signal_gate = abs(t_iter(0, :)) / max(maxval(abs(t_iter(0, :))), 1.0e-12_real32)
      do step_idx = 1, model%num_corrections
         do i = 1, ge
            corr_in(1:w) = x_final(:, i)
            corr_in(w + 1) = t_iter(step_idx - 1, i)
            corr1_pre(:, step_idx, i) = matmul(model%corr1_w, corr_in) + model%corr1_b
            corr1_act(:, step_idx, i) = silu(corr1_pre(:, step_idx, i))
            corr2_pre(:, step_idx, i) = matmul(model%corr2_w, corr1_act(:, step_idx, i)) + model%corr2_b
            corr2_act(:, step_idx, i) = silu(corr2_pre(:, step_idx, i))
            corr_delta(step_idx, i) = dot_product(model%corr3_w, corr2_act(:, step_idx, i)) + model%corr3_b
            t_iter(step_idx, i) = t_iter(step_idx - 1, i) + model%step_sizes(step_idx) * corr_delta(step_idx, i) * signal_gate(i)
         end do
      end do

      relax_strength = softplus(model%relax_log_strength)
      do i = 1, g
         relax_dir(i) = model%bc_left_norm + (model%bc_right_norm - model%bc_left_norm) * model%xi(i) - t_n(i)
         gate_in(1:w) = x_final(:, i + 1)
         gate_in(w + 1) = t_iter(model%num_corrections, i + 1)
         gate1_pre(:, i) = matmul(model%gate1_w, gate_in) + model%gate1_b
         gate1_act(:, i) = silu(gate1_pre(:, i))
         gate(i) = sigmoid(dot_product(model%gate2_w, gate1_act(:, i)) + model%gate2_b)
      end do

      loss_scale = 2.0_real32 / real(g, real32)
      do i = 1, g
         dy(i) = loss_scale * (pred(i) - target(i))
      end do

      dx = 0.0_real32
      do i = 1, g
         idx = i + 1
         d_gated = dy(i) * model%boundary_mask(i)

         d_gate_pre = d_gated * relax_strength * relax_dir(i) * gate(i) * (1.0_real32 - gate(i))
         grad%gate2_w = grad%gate2_w + d_gate_pre * gate1_act(:, i)
         grad%gate2_b = grad%gate2_b + d_gate_pre
         grad%relax_log_strength = grad%relax_log_strength + d_gated * gate(i) * relax_dir(i) * sigmoid(model%relax_log_strength)

         d_gate_h1 = d_gate_pre * model%gate2_w
         d_gate_h1_pre = d_gate_h1 * dsilu(gate1_pre(:, i))
         do c = 1, gh
            grad%gate1_w(c, 1:w) = grad%gate1_w(c, 1:w) + d_gate_h1_pre(c) * x_final(:, idx)
            grad%gate1_w(c, w + 1) = grad%gate1_w(c, w + 1) + d_gate_h1_pre(c) * t_iter(model%num_corrections, idx)
         end do
         grad%gate1_b = grad%gate1_b + d_gate_h1_pre
         do c = 1, w
            dx(c, idx) = dx(c, idx) + dot_product(model%gate1_w(:, c), d_gate_h1_pre)
         end do
         d_tinc = d_gated + dot_product(model%gate1_w(:, w + 1), d_gate_h1_pre)

         do step_idx = model%num_corrections, 1, -1
            grad%step_sizes(step_idx) = grad%step_sizes(step_idx) + d_tinc * corr_delta(step_idx, idx) * signal_gate(idx)
            grad%corr3_w = grad%corr3_w + d_tinc * model%step_sizes(step_idx) * signal_gate(idx) * corr2_act(:, step_idx, idx)
            grad%corr3_b = grad%corr3_b + d_tinc * model%step_sizes(step_idx) * signal_gate(idx)

            d_h2 = d_tinc * model%step_sizes(step_idx) * signal_gate(idx) * model%corr3_w
            d_h2_pre = d_h2 * dsilu(corr2_pre(:, step_idx, idx))
            do c = 1, ch
               grad%corr2_w(c, :) = grad%corr2_w(c, :) + d_h2_pre(c) * corr1_act(:, step_idx, idx)
               grad%corr2_b(c) = grad%corr2_b(c) + d_h2_pre(c)
            end do

            d_h1 = matmul(transpose(model%corr2_w), d_h2_pre)
            d_h1_pre = d_h1 * dsilu(corr1_pre(:, step_idx, idx))
            do c = 1, w
               grad%corr1_w(c, 1:w) = grad%corr1_w(c, 1:w) + d_h1_pre(c) * x_final(:, idx)
               grad%corr1_w(c, w + 1) = grad%corr1_w(c, w + 1) + d_h1_pre(c) * t_iter(step_idx - 1, idx)
               grad%corr1_b(c) = grad%corr1_b(c) + d_h1_pre(c)
            end do
            do c = 1, w
               dx(c, idx) = dx(c, idx) + dot_product(model%corr1_w(:, c), d_h1_pre)
            end do
            d_tinc = d_tinc + dot_product(model%corr1_w(:, w + 1), d_h1_pre)
         end do

         raw_a_grad = d_tinc * sec_diff_ext(idx) * bound_a * (1.0_real32 - tanh(raw_a(idx)) ** 2)
         raw_b_grad = d_tinc * diff_ext(idx) * bound_b * (1.0_real32 - tanh(raw_b(idx)) ** 2)
         grad%coeff3_w = grad%coeff3_w + raw_a_grad * coeff2_act(:, idx)
         grad%coeff3_b = grad%coeff3_b + raw_a_grad
         grad%coeff4_w = grad%coeff4_w + raw_b_grad * coeff2_act(:, idx)
         grad%coeff4_b = grad%coeff4_b + raw_b_grad

         d_h2 = raw_a_grad * model%coeff3_w + raw_b_grad * model%coeff4_w
         d_h2_pre = d_h2 * dsilu(coeff2_pre(:, idx))
         do c = 1, ch
            grad%coeff2_w(c, :) = grad%coeff2_w(c, :) + d_h2_pre(c) * coeff1_act(:, idx)
            grad%coeff2_b(c) = grad%coeff2_b(c) + d_h2_pre(c)
         end do

         d_h1 = matmul(transpose(model%coeff2_w), d_h2_pre)
         d_h1_pre = d_h1 * dsilu(coeff1_pre(:, idx))
         do c = 1, w
            grad%coeff1_w(c, 1:w) = grad%coeff1_w(c, 1:w) + d_h1_pre(c) * x_final(:, idx)
            grad%coeff1_w(c, w + 1) = grad%coeff1_w(c, w + 1) + d_h1_pre(c) * fo_log
            grad%coeff1_w(c, w + 2) = grad%coeff1_w(c, w + 2) + d_h1_pre(c) * model%ve_scalar
            grad%coeff1_b(c) = grad%coeff1_b(c) + d_h1_pre(c)
         end do
         do c = 1, w
            dx(c, idx) = dx(c, idx) + dot_product(model%coeff1_w(:, c), d_h1_pre)
         end do
      end do

      do b = model%num_blocks, 1, -1
         allocate(dkernel_x(m, w, ge), dkernels(m, ge, ge), draw(m, ge, ge), dact(w, ge), dz(w, ge), dnorm(w, ge))
         dkernel_x = 0.0_real32
         dkernels = 0.0_real32
         draw = 0.0_real32
         dact = 0.0_real32
         dz = 0.0_real32
         dnorm = 0.0_real32
         xmean_grad = 0.0_real32

         do k = 1, m
            draw(k, 1:w, 1:w) = matmul(caches(b)%kernel_x(k, :, :), transpose(dx))
            grad%wt_log_amp(k, :, :, b) = grad%wt_log_amp(k, :, :, b) + draw(k, 1:w, 1:w) * model%amp_sharpness * &
                 (caches(b)%amp(k, :, :) / model%max_amp) * (1.0_real32 - caches(b)%amp(k, :, :) / model%max_amp) * model%max_amp * &
                 cos(model%wt_phase(k, :, :, b))
            grad%wt_phase(k, :, :, b) = grad%wt_phase(k, :, :, b) - draw(k, 1:w, 1:w) * caches(b)%amp(k, :, :) * &
                 sin(model%wt_phase(k, :, :, b))
            dkernel_x(k, :, :) = dkernel_x(k, :, :) + matmul(caches(b)%weights(k, :, :), dx)
         end do

         do k = 1, m
            dact = dact + matmul(dkernel_x(k, :, :), caches(b)%kernels(k, :, :))
            dkernels(k, :, :) = dkernels(k, :, :) + matmul(transpose(dkernel_x(k, :, :)), caches(b)%act_out)
         end do

         do k = 1, m
            dpole = 0.0_real32
            do i = 1, ge
                   draw(k, i, :) = dkernels(k, i, :) * model%spec_filter(k) / caches(b)%row_sum(k, i)
               dpole = dpole - sum(draw(k, i, :) * model%dist(i, :) * caches(b)%raw_kernels(k, i, :))
            end do
            pole_preact = model%log_poles(k, b) + model%pole_offset_scale * caches(b)%pole_offset(k)
                   sig = sigmoid(pole_preact)
                   if (caches(b)%poles(k) <= model%pole_min .or. caches(b)%poles(k) >= model%pole_max) sig = 0.0_real32
            grad%log_poles(k, b) = grad%log_poles(k, b) + dpole * sig
            pole_offset_grad = dpole * sig * model%pole_offset_scale * (1.0_real32 - caches(b)%pole_offset(k) ** 2)
            grad%pole_mlp_w(k, :, b) = grad%pole_mlp_w(k, :, b) + pole_offset_grad * caches(b)%x_mean
            grad%pole_mlp_b(k, b) = grad%pole_mlp_b(k, b) + pole_offset_grad
            xmean_grad = xmean_grad + pole_offset_grad * model%pole_mlp_w(k, :, b)
         end do

         do c = 1, w
            dact(c, :) = dact(c, :) + xmean_grad(c) / real(ge, real32)
         end do

         dz = dact * dsilu(caches(b)%pw_out)
         grad%pw_w(:, :, b) = grad%pw_w(:, :, b) + matmul(dz, transpose(caches(b)%pw_in))
         grad%pw_b(:, b) = grad%pw_b(:, b) + sum(dz, dim=2)
         dnorm = dnorm + matmul(transpose(model%pw_w(:, :, b)), dz)

         dprev = 0.0_real32
         do c = 1, w
            centered_sum = sum(dnorm(c, :))
            xhat_sum = sum(dnorm(c, :) * caches(b)%x_hat(c, :))
            do i = 1, ge
               dprev(c, i) = dprev(c, i) + caches(b)%inv_std(c) / real(ge, real32) * &
                    (real(ge, real32) * dnorm(c, i) - centered_sum - caches(b)%x_hat(c, i) * xhat_sum)
            end do
         end do

         dx = dprev
         deallocate(dkernel_x, dkernels, draw, dact, dz, dnorm)
      end do

      do c = 1, w
         grad%proj_b(c) = sum(dx(c, :))
         grad%proj_w(c, 1) = sum(dx(c, :) * t_n_ext)
      end do

      call free_caches(caches, x0, x_final)
      deallocate(pred, dx, dprev, t_n, t_nm1, t_n_ext, t_nm1_ext, sec_diff_ext, diff_ext, signal_gate, dy, coeff1_pre, coeff1_act, coeff2_pre, coeff2_act, raw_a, raw_b, &
           t_iter, corr_delta, corr1_pre, corr1_act, corr2_pre, corr2_act, gate1_pre, gate1_act, gate, relax_dir, coeff_in, corr_in, gate_in, d_h1, d_h1_pre, d_h2, d_h2_pre, &
           d_gate_h1, d_gate_h1_pre, xmean_grad)
  end subroutine compute_gradients

  subroutine adam_update_1d(param, m_buf, v_buf, grad, model, bc1, bc2)
    real(real32), intent(inout) :: param(:), m_buf(:), v_buf(:)
    real(real32), intent(in) :: grad(:)
    type(custom_lno_model_type), intent(in) :: model
    real(real32), intent(in) :: bc1, bc2
    m_buf = model%beta1 * m_buf + (1.0_real32 - model%beta1) * grad
    v_buf = model%beta2 * v_buf + (1.0_real32 - model%beta2) * grad * grad
    param = param * (1.0_real32 - model%lr * model%weight_decay)
    param = param - model%lr * (m_buf / bc1) / (sqrt(v_buf / bc2) + EPS_ADAM)
  end subroutine adam_update_1d

  subroutine adam_update_2d(param, m_buf, v_buf, grad, model, bc1, bc2)
    real(real32), intent(inout) :: param(:,:), m_buf(:,:), v_buf(:,:)
    real(real32), intent(in) :: grad(:,:)
    type(custom_lno_model_type), intent(in) :: model
    real(real32), intent(in) :: bc1, bc2
    m_buf = model%beta1 * m_buf + (1.0_real32 - model%beta1) * grad
    v_buf = model%beta2 * v_buf + (1.0_real32 - model%beta2) * grad * grad
    param = param * (1.0_real32 - model%lr * model%weight_decay)
    param = param - model%lr * (m_buf / bc1) / (sqrt(v_buf / bc2) + EPS_ADAM)
  end subroutine adam_update_2d

  subroutine adam_update_3d(param, m_buf, v_buf, grad, model, bc1, bc2)
    real(real32), intent(inout) :: param(:,:,:), m_buf(:,:,:), v_buf(:,:,:)
    real(real32), intent(in) :: grad(:,:,:)
    type(custom_lno_model_type), intent(in) :: model
    real(real32), intent(in) :: bc1, bc2
    m_buf = model%beta1 * m_buf + (1.0_real32 - model%beta1) * grad
    v_buf = model%beta2 * v_buf + (1.0_real32 - model%beta2) * grad * grad
    param = param * (1.0_real32 - model%lr * model%weight_decay)
    param = param - model%lr * (m_buf / bc1) / (sqrt(v_buf / bc2) + EPS_ADAM)
  end subroutine adam_update_3d

  subroutine adam_update_4d(param, m_buf, v_buf, grad, model, bc1, bc2)
    real(real32), intent(inout) :: param(:,:,:,:), m_buf(:,:,:,:), v_buf(:,:,:,:)
    real(real32), intent(in) :: grad(:,:,:,:)
    type(custom_lno_model_type), intent(in) :: model
    real(real32), intent(in) :: bc1, bc2
    m_buf = model%beta1 * m_buf + (1.0_real32 - model%beta1) * grad
    v_buf = model%beta2 * v_buf + (1.0_real32 - model%beta2) * grad * grad
    param = param * (1.0_real32 - model%lr * model%weight_decay)
    param = param - model%lr * (m_buf / bc1) / (sqrt(v_buf / bc2) + EPS_ADAM)
  end subroutine adam_update_4d

   subroutine free_caches(caches, x0, x_final)
    type(block_cache_type), allocatable, intent(inout) :: caches(:)
    real(real32), allocatable, intent(inout) :: x0(:,:)
      real(real32), allocatable, intent(inout) :: x_final(:,:)
    integer :: b
    if (allocated(caches)) then
       do b = 1, size(caches)
          if (allocated(caches(b)%x_in)) deallocate(caches(b)%x_in)
          if (allocated(caches(b)%x_hat)) deallocate(caches(b)%x_hat)
          if (allocated(caches(b)%mu)) deallocate(caches(b)%mu)
          if (allocated(caches(b)%inv_std)) deallocate(caches(b)%inv_std)
          if (allocated(caches(b)%pw_in)) deallocate(caches(b)%pw_in)
          if (allocated(caches(b)%pw_out)) deallocate(caches(b)%pw_out)
          if (allocated(caches(b)%act_out)) deallocate(caches(b)%act_out)
          if (allocated(caches(b)%kernel_x)) deallocate(caches(b)%kernel_x)
          if (allocated(caches(b)%kernels)) deallocate(caches(b)%kernels)
          if (allocated(caches(b)%raw_kernels)) deallocate(caches(b)%raw_kernels)
          if (allocated(caches(b)%row_sum)) deallocate(caches(b)%row_sum)
          if (allocated(caches(b)%poles)) deallocate(caches(b)%poles)
          if (allocated(caches(b)%x_mean)) deallocate(caches(b)%x_mean)
          if (allocated(caches(b)%pole_offset)) deallocate(caches(b)%pole_offset)
          if (allocated(caches(b)%amp)) deallocate(caches(b)%amp)
          if (allocated(caches(b)%weights)) deallocate(caches(b)%weights)
       end do
       deallocate(caches)
    end if
    if (allocated(x0)) deallocate(x0)
      if (allocated(x_final)) deallocate(x_final)
  end subroutine free_caches

   subroutine compute_rollout_window_gradients(model, trajectory, start_step, rollout_steps, alpha_scalar, tau_scalar, bc_left_norm, bc_right_norm, dt, dx, grad, loss)
    type(custom_lno_model_type), intent(inout) :: model
    real(real32), intent(in) :: trajectory(:,:)
    integer, intent(in) :: start_step, rollout_steps
      real(real32), intent(in) :: alpha_scalar, tau_scalar, bc_left_norm, bc_right_norm, dt, dx
    type(grad_accum_type), intent(inout) :: grad
    real(real32), intent(out) :: loss

    integer :: g, k, available_steps
    real(real32), allocatable :: prev_state(:), curr_state(:), input_vec(:), pred_inc(:), target_inc(:)
    real(real32) :: step_loss
    type(grad_accum_type) :: step_grad

    g = model%grid_size
    available_steps = min(rollout_steps, size(trajectory, 2) - start_step - 1)
    call zero_gradients(grad)
    loss = 0.0_real32
    if (available_steps < 1) return

   call set_runtime_conditions(model, bc_left_norm, bc_right_norm, alpha_scalar, tau_scalar, dt, dx)

    allocate(prev_state(g), curr_state(g), input_vec(2 * g), pred_inc(g), target_inc(g))
    call allocate_gradients(model, step_grad)

    prev_state = trajectory(:, start_step)
    curr_state = trajectory(:, start_step + 1)

    do k = 1, available_steps
       input_vec(1:g) = curr_state
       input_vec(g + 1:2 * g) = prev_state
       target_inc = trajectory(:, start_step + k + 1) - curr_state

       call compute_gradients(model, input_vec, target_inc, step_grad, step_loss)
       call add_gradients(grad, step_grad)
       loss = loss + step_loss

       call custom_lno_predict(model, input_vec, pred_inc)
       prev_state = curr_state
       curr_state = curr_state + pred_inc
    end do

    call scale_gradients(grad, 1.0_real32 / real(available_steps, real32))
    loss = loss / real(available_steps, real32)

    call deallocate_gradients(step_grad)
    deallocate(prev_state, curr_state, input_vec, pred_inc, target_inc)
  end subroutine compute_rollout_window_gradients

   subroutine compute_rollout_validation_loss(model, trajectory, rollout_steps, alpha_scalar, tau_scalar, bc_left_norm, bc_right_norm, dt, dx, loss)
      type(custom_lno_model_type), intent(inout) :: model
    real(real32), intent(in) :: trajectory(:,:)
    integer, intent(in) :: rollout_steps
      real(real32), intent(in) :: alpha_scalar, tau_scalar, bc_left_norm, bc_right_norm, dt, dx
    real(real32), intent(out) :: loss

    integer :: g, k, available_steps
    real(real32), allocatable :: prev_state(:), curr_state(:), input_vec(:), pred_inc(:), target_inc(:)

    g = model%grid_size
    available_steps = min(rollout_steps, size(trajectory, 2) - 2)
    loss = 0.0_real32
    if (available_steps < 1) return

   call set_runtime_conditions(model, bc_left_norm, bc_right_norm, alpha_scalar, tau_scalar, dt, dx)

    allocate(prev_state(g), curr_state(g), input_vec(2 * g), pred_inc(g), target_inc(g))
    prev_state = trajectory(:, 1)
    curr_state = trajectory(:, 2)

    do k = 1, available_steps
       input_vec(1:g) = curr_state
       input_vec(g + 1:2 * g) = prev_state
       call custom_lno_predict(model, input_vec, pred_inc)
       target_inc = trajectory(:, k + 2) - curr_state
       loss = loss + sum((pred_inc - target_inc) ** 2) / real(g, real32)
       prev_state = curr_state
       curr_state = curr_state + pred_inc
    end do

    loss = loss / real(available_steps, real32)
    deallocate(prev_state, curr_state, input_vec, pred_inc, target_inc)
  end subroutine compute_rollout_validation_loss

   subroutine custom_lno_train(model, train_inputs, train_targets, val_inputs, val_targets, num_epochs, batch_size, alpha_scalar, dt, dx, &
          train_tau, train_bc_left, train_bc_right, val_tau, val_bc_left, val_bc_right, &
          train_trajectories, train_trajectory_tau, train_trajectory_bc_left, train_trajectory_bc_right, &
          val_trajectories, val_trajectory_tau, val_trajectory_bc_left, val_trajectory_bc_right, &
        rollout_weight, rollout_steps, rollout_steps_min, rollout_warmup_epochs, use_cosine_schedule, &
        lambda_steady_state, steady_state_every_n_batches)
    type(custom_lno_model_type), intent(inout) :: model
    real(real32), intent(in) :: train_inputs(:,:), train_targets(:,:)
    real(real32), intent(in) :: val_inputs(:,:), val_targets(:,:)
    integer, intent(in) :: num_epochs, batch_size
      real(real32), intent(in) :: alpha_scalar, dt, dx
      real(real32), intent(in) :: train_tau(:), train_bc_left(:), train_bc_right(:)
      real(real32), intent(in) :: val_tau(:), val_bc_left(:), val_bc_right(:)
      real(real32), intent(in), optional :: train_trajectories(:,:,:), val_trajectories(:,:,:)
      real(real32), intent(in), optional :: train_trajectory_tau(:), train_trajectory_bc_left(:), train_trajectory_bc_right(:)
      real(real32), intent(in), optional :: val_trajectory_tau(:), val_trajectory_bc_left(:), val_trajectory_bc_right(:)
      real(real32), intent(in), optional :: rollout_weight, lambda_steady_state
      integer, intent(in), optional :: rollout_steps, rollout_steps_min, rollout_warmup_epochs, steady_state_every_n_batches
      logical, intent(in), optional :: use_cosine_schedule

      integer :: epoch, n_train, n_val, i, j, start_idx, end_idx, idx, batch_count, sample_idx, traj_idx
      integer :: batch_number, steady_state_stride, steady_state_idx
      integer :: current_rollout_steps, current_rollout_steps_min, max_rollout_start, rollout_start, current_rollout_warmup_epochs
    integer, allocatable :: perm(:)
    real(real32) :: train_loss, val_loss, batch_loss, sample_loss, rollout_loss, current_rollout_weight
      real(real32) :: val_rollout_loss, traj_rollout_loss, r, warmup_linear, warmup_frac, initial_lr, current_lambda_steady_state
      logical :: use_rollout_training, use_rollout_validation, enable_cosine_schedule
    type(grad_accum_type) :: batch_grad
    type(grad_accum_type) :: sample_grad
      real(real32), allocatable :: steady_state_input(:), steady_state_target(:)

    n_train = size(train_inputs, 2)
    n_val = size(val_inputs, 2)
    allocate(perm(n_train))
    call allocate_gradients(model, batch_grad)
    call allocate_gradients(model, sample_grad)

    enable_cosine_schedule = .true.
    if (present(use_cosine_schedule)) enable_cosine_schedule = use_cosine_schedule
    initial_lr = model%lr
   current_lambda_steady_state = 0.0_real32
   if (present(lambda_steady_state)) current_lambda_steady_state = max(0.0_real32, lambda_steady_state)
   steady_state_stride = 4
   if (present(steady_state_every_n_batches)) steady_state_stride = max(1, steady_state_every_n_batches)
    current_rollout_steps_min = 1
    if (present(rollout_steps_min)) current_rollout_steps_min = max(1, rollout_steps_min)
    current_rollout_warmup_epochs = 1
    if (present(rollout_warmup_epochs)) current_rollout_warmup_epochs = max(1, rollout_warmup_epochs)

   allocate(steady_state_input(2 * model%grid_size), steady_state_target(model%grid_size))
   steady_state_target = 0.0_real32

    use_rollout_training = present(train_trajectories) .and. present(train_trajectory_tau) .and. present(train_trajectory_bc_left) .and. &
         present(train_trajectory_bc_right) .and. present(rollout_weight) .and. present(rollout_steps)
    use_rollout_validation = present(val_trajectories) .and. present(val_trajectory_tau) .and. present(val_trajectory_bc_left) .and. &
         present(val_trajectory_bc_right) .and. present(rollout_steps)

    do epoch = 1, num_epochs
       if (enable_cosine_schedule) model%lr = cosine_annealed_lr(initial_lr, epoch - 1, num_epochs, 0.01_real32)
       call random_permutation(perm)
       train_loss = 0.0_real32
       if (use_rollout_training) then
          warmup_linear = min(1.0_real32, real(epoch, real32) / real(max(1, current_rollout_warmup_epochs), real32))
          warmup_frac = warmup_linear * warmup_linear
          current_rollout_steps = current_rollout_steps_min + int(warmup_frac * real(max(0, rollout_steps - current_rollout_steps_min), real32))
          current_rollout_steps = max(1, min(min(rollout_steps, size(train_trajectories, 2) - 2), current_rollout_steps))
          current_rollout_weight = rollout_weight * warmup_frac
          max_rollout_start = max(1, size(train_trajectories, 2) - current_rollout_steps - 1)
       else
          current_rollout_steps = 0
          current_rollout_weight = 0.0_real32
          max_rollout_start = 0
       end if
       do start_idx = 1, n_train, max(1, batch_size)
          end_idx = min(n_train, start_idx + max(1, batch_size) - 1)
          batch_number = 1 + (start_idx - 1) / max(1, batch_size)
          batch_count = end_idx - start_idx + 1
          batch_loss = 0.0_real32
          call zero_gradients(batch_grad)
          do sample_idx = start_idx, end_idx
             idx = perm(sample_idx)
             call set_runtime_conditions(model, train_bc_left(idx), train_bc_right(idx), alpha_scalar, train_tau(idx), dt, dx)
             call compute_gradients(model, train_inputs(:, idx), train_targets(:, idx), sample_grad, sample_loss)
             call add_gradients(batch_grad, sample_grad)
             batch_loss = batch_loss + sample_loss
          end do
          if (current_lambda_steady_state > 0.0_real32 .and. mod(batch_number - 1, steady_state_stride) == 0) then
             steady_state_idx = perm(start_idx)
             call build_steady_state_input(model, train_bc_left(steady_state_idx), train_bc_right(steady_state_idx), steady_state_input)
             call set_runtime_conditions(model, train_bc_left(steady_state_idx), train_bc_right(steady_state_idx), alpha_scalar, train_tau(steady_state_idx), dt, dx)
             call compute_gradients(model, steady_state_input, steady_state_target, sample_grad, sample_loss)
             call scale_gradients(sample_grad, current_lambda_steady_state)
             call add_gradients(batch_grad, sample_grad)
             batch_loss = batch_loss + current_lambda_steady_state * sample_loss
          end if
          if (use_rollout_training .and. current_rollout_weight > 0.0_real32) then
             call random_number(r)
             traj_idx = 1 + int(r * real(size(train_trajectories, 3), real32))
             traj_idx = max(1, min(size(train_trajectories, 3), traj_idx))
             call random_number(r)
             rollout_start = 1 + int(r * real(max_rollout_start, real32))
             rollout_start = max(1, min(max_rollout_start, rollout_start))
             call compute_rollout_window_gradients(model, train_trajectories(:, :, traj_idx), rollout_start, current_rollout_steps, alpha_scalar, &
                  train_trajectory_tau(traj_idx), train_trajectory_bc_left(traj_idx), train_trajectory_bc_right(traj_idx), dt, dx, sample_grad, rollout_loss)
             call scale_gradients(sample_grad, current_rollout_weight)
             call add_gradients(batch_grad, sample_grad)
             batch_loss = batch_loss + current_rollout_weight * rollout_loss
          end if
          call scale_gradients(batch_grad, 1.0_real32 / real(batch_count, real32))
          call apply_gradients(model, batch_grad)
          train_loss = train_loss + batch_loss
       end do
       train_loss = train_loss / real(n_train, real32)

       val_loss = 0.0_real32
       do j = 1, n_val
          block
             real(real32) :: pred_local(model%grid_size)
             call set_runtime_conditions(model, val_bc_left(j), val_bc_right(j), alpha_scalar, val_tau(j), dt, dx)
             call custom_lno_predict(model, val_inputs(:, j), pred_local)
             val_loss = val_loss + sum((pred_local - val_targets(:, j)) ** 2) / real(model%grid_size, real32)
          end block
       end do
       val_loss = val_loss / real(n_val, real32)
       if (use_rollout_validation) then
          val_rollout_loss = 0.0_real32
          do traj_idx = 1, size(val_trajectories, 3)
             call compute_rollout_validation_loss(model, val_trajectories(:, :, traj_idx), rollout_steps, alpha_scalar, &
                  val_trajectory_tau(traj_idx), val_trajectory_bc_left(traj_idx), val_trajectory_bc_right(traj_idx), dt, dx, traj_rollout_loss)
             val_rollout_loss = val_rollout_loss + traj_rollout_loss
          end do
          val_rollout_loss = val_rollout_loss / real(size(val_trajectories, 3), real32)
          write(*,'(A,I0,A,ES12.4,A,ES12.4,A,ES12.4)') 'epoch=', epoch, ', train_loss=', train_loss, &
               ', val_loss=', val_loss, ', val_rollout_loss=', val_rollout_loss
       else
          write(*,'(A,I0,A,ES12.4,A,ES12.4)') 'epoch=', epoch, ', train_loss=', train_loss, ', val_loss=', val_loss
       end if
    end do

       call deallocate_gradients(sample_grad)
       call deallocate_gradients(batch_grad)
     deallocate(steady_state_input, steady_state_target)
      deallocate(perm)
  end subroutine custom_lno_train

  subroutine random_permutation(perm)
    integer, intent(out) :: perm(:)
    integer :: i, j, tmp
    real(real32) :: r
    do i = 1, size(perm)
       perm(i) = i
    end do
    do i = size(perm), 2, -1
       call random_number(r)
       j = 1 + int(r * real(i, real32))
       j = max(1, min(i, j))
       tmp = perm(i)
       perm(i) = perm(j)
       perm(j) = tmp
    end do
  end subroutine random_permutation

   subroutine custom_lno_rollout(model, init_state, rollout_steps, history, temp_ref, delta_t, alpha_scalar, tau_scalar, bc_left_norm, bc_right_norm, dt, dx)
      type(custom_lno_model_type), intent(inout) :: model
    real(real32), intent(in) :: init_state(:)
    integer, intent(in) :: rollout_steps
    real(real32), intent(out) :: history(:,:)
    real(real32), intent(in) :: temp_ref, delta_t
      real(real32), intent(in) :: alpha_scalar, tau_scalar, bc_left_norm, bc_right_norm, dt, dx

    real(real32), allocatable :: prev_state(:), curr_state(:), input_vec(:), pred(:), curr_scaled(:), prev_scaled(:)
    integer :: i, g

    g = model%grid_size
      call set_runtime_conditions(model, bc_left_norm, bc_right_norm, alpha_scalar, tau_scalar, dt, dx)
    allocate(prev_state(g), curr_state(g), input_vec(2 * g), pred(g), curr_scaled(g), prev_scaled(g))
    prev_state = init_state
    curr_state = init_state
    history(1, :) = curr_state
   do i = 1, rollout_steps
       curr_scaled = (curr_state - temp_ref) / delta_t
       prev_scaled = (prev_state - temp_ref) / delta_t
       input_vec(1:g) = curr_scaled
       input_vec(g + 1:2 * g) = prev_scaled
       call custom_lno_predict(model, input_vec, pred)
       prev_state = curr_state
       curr_state = temp_ref + delta_t * (curr_scaled + pred)
       history(i + 1, :) = curr_state
    end do
    deallocate(prev_state, curr_state, input_vec, pred, curr_scaled, prev_scaled)
  end subroutine custom_lno_rollout

end module custom_lno_trainable