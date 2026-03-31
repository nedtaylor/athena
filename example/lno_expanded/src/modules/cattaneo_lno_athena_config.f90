module cattaneo_lno_athena_config
  !! Configuration types for the Athena-backed Cattaneo-LNO scaffold.
  !!
  !! This module stores the model hyperparameters required to mirror the
  !! Python architecture layout while keeping the Athena dependency untouched.
  use coreutils, only: real32
  implicit none

  private

  public :: cattaneo_lno_config_type

  type :: cattaneo_lno_config_type
     !! Configuration for the Athena-backed Cattaneo-LNO scaffold.
     integer :: grid_size = 256
     !! Number of interior grid points.
      real(real32) :: domain_length = 1.0_real32
      !! Physical domain length used for dimensionless coordinates.
      real(real32) :: alpha_ref = 1.0e-4_real32
      !! Reference thermal diffusivity used by the dimensionless scaler.
      real(real32) :: temp_ref = 200.0_real32
      !! Reference temperature for dimensionless conversion.
      real(real32) :: delta_temp = 100.0_real32
      !! Characteristic temperature scale for dimensionless conversion.
      real(real32) :: tau_ref = 1.0e-9_real32
      !! Reference relaxation time retained for scaler parity.
      real(real32) :: fo_min = 1.0e-6_real32
      !! Lower clamp for the Fourier-number feature field.
      real(real32) :: fo_max = 500.0_real32
      !! Upper clamp for the Fourier-number feature field.
     integer :: modes = 16
     !! Number of Laplace modes in the structural stand-in operator.
     integer :: width = 64
     !! Backbone channel width.
     integer :: num_no_layers = 4
     !! Number of LNO blocks in the predictor backbone.
     character(len=16) :: activation = "swish"
     !! Activation name used for Athena-backed projections.
     integer :: timestep_jump = 1
     !! Number of FDM steps represented by one neural step.
     logical :: use_ghost_cells = .true.
     !! Whether to include two boundary ghost cells in the working grid.
     integer :: history_len = 4
     !! Number of temperature history frames expected by the Python model.
     integer :: temporal_channels = 32
     !! Reserved temporal feature width from the Python design.
     integer :: local_conv_layers = 2
     !! Reserved local-path depth from the Python design.
     integer :: num_corrections = 3
     !! Number of iterative corrector refinement steps.
     logical :: use_recurrent_memory = .false.
     !! Whether recurrent memory is enabled.
     integer :: memory_channels = 32
     !! Hidden width of the recurrent memory path.
     integer :: num_internal_steps = 1
     !! Number of learned internal substeps per macro step.
     character(len=32) :: spectral_filter = "exponential"
     !! Reserved filter descriptor from the Python design.
     real(real32) :: filter_strength = 4.0_real32
     !! Filter strength from the Python design.
     real(real32) :: max_amp = 1.0_real32
     !! Upper bound of the polar spectral amplitude.
     real(real32) :: amp_sharpness = 1.0_real32
     !! Sharpness used inside the sigmoid-bounded amplitude mapping.
     real(real32) :: pole_offset_scale = 0.1_real32
     !! Scale factor applied to the data-dependent pole-offset MLP.
     real(real32) :: pole_min = 0.1_real32
     !! Minimum allowed positive pole value.
     real(real32) :: pole_max = 100.0_real32
     !! Maximum allowed positive pole value.
     logical :: use_causal_mask = .true.
     !! Whether the Python-style causal mask is enabled.
     real(real32) :: causal_safety = 1.0_real32
     !! Safety factor applied to the characteristic speed in the causal mask.
     real(real32) :: cfl_threshold = 0.5_real32
     !! Stability threshold used by the Python forward path.
     real(real32) :: learning_rate = 1.0e-4_real32
     !! Stable rollout baseline learning rate for swish-based Laplace layers.
   contains
     procedure :: extended_grid
     !! Return the working grid size including ghost cells when enabled.
  end type cattaneo_lno_config_type

contains

  integer function extended_grid(this)
    !! Return the model grid size seen by Athena-backed layers.
    class(cattaneo_lno_config_type), intent(in) :: this

    extended_grid = this%grid_size
    if (this%use_ghost_cells) extended_grid = extended_grid + 2
  end function extended_grid

end module cattaneo_lno_athena_config