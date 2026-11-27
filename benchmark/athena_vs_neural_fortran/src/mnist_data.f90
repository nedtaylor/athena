module mnist_data
  !! Reads the plain-text MNIST files shared with the Athena example.
  !!
  !! Each line is `label,pixel(1),...,pixel(784)`. Pixels are normalised to
  !! [0,1] exactly as the Athena MNIST example does.
  use iso_fortran_env, only: real32
  implicit none
  private
  public :: load_mnist

contains

  subroutine load_mnist(dir, n_train, train_x, train_y, test_x, test_y)
    !! Load the first `n_train` training rows and all test rows.
    character(len=*), intent(in) :: dir
    integer, intent(in) :: n_train
    real(real32), allocatable, intent(out) :: train_x(:,:), test_x(:,:)
    integer, allocatable, intent(out) :: train_y(:), test_y(:)

    integer, parameter :: pixels = 784

    allocate(train_x(pixels, n_train), train_y(n_train))
    call read_split(trim(dir)//'/MNIST_train.txt', n_train, train_x, train_y)

    allocate(test_x(pixels, 10000), test_y(10000))
    call read_split(trim(dir)//'/MNIST_test.txt', 10000, test_x, test_y)
  end subroutine load_mnist

  subroutine read_split(path, n, x, y)
    character(len=*), intent(in) :: path
    integer, intent(in) :: n
    real(real32), intent(out) :: x(:,:)
    integer, intent(out) :: y(:)
    real(real32) :: row(784)
    integer :: i, j, label, unit

    open(newunit=unit, file=path, status='old', action='read')
    do i = 1, n
       read(unit, *) label, (row(j), j=1, 784)
       if(label < 0 .or. label > 9) error stop 'invalid MNIST label'
       y(i) = label
       x(:,i) = row / 255._real32
    end do
    close(unit)
  end subroutine read_split

end module mnist_data
