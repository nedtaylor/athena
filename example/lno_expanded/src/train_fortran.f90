program train_fortran
  use cattaneo_lno_athena, only: cattaneo_lno_config_type, cattaneo_lno_athena_type
  implicit none

  type(cattaneo_lno_config_type) :: config
  type(cattaneo_lno_athena_type) :: model

  config%grid_size = 256
  config%modes = 16
  config%width = 64
  config%num_no_layers = 4
  config%activation = "swish"
  config%timestep_jump = 200
  config%use_ghost_cells = .true.
  config%history_len = 4
  config%temporal_channels = 32
  config%local_conv_layers = 2
  config%num_corrections = 3
  config%use_recurrent_memory = .true.
  config%memory_channels = 32
  config%num_internal_steps = 1
  config%learning_rate = 1.0e-4

  call model%init(config)
  call model%print_summary()
end program train_fortran