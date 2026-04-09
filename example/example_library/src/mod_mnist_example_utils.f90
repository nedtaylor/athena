module mnist_example_utils
  use constants_mnist, only: real32
  implicit none

  private

  public :: limit_mnist_dataset

contains

  subroutine limit_mnist_dataset(images, labels, max_samples)
    implicit none
    real(real32), allocatable, intent(inout), dimension(:,:,:,:) :: images
    integer, allocatable, intent(inout), dimension(:) :: labels
    integer, intent(in) :: max_samples

    real(real32), allocatable, dimension(:,:,:,:) :: image_buffer
    integer, allocatable, dimension(:) :: label_buffer
    integer :: sample_count

    sample_count = min(size(images, 4), max_samples)
    if(sample_count.eq.size(images, 4)) return

    allocate(image_buffer( &
         size(images, 1), size(images, 2), size(images, 3), sample_count))
    image_buffer = images(:,:,:,1:sample_count)
    call move_alloc(image_buffer, images)

    allocate(label_buffer(sample_count))
    label_buffer = labels(1:sample_count)
    call move_alloc(label_buffer, labels)
  end subroutine limit_mnist_dataset

end module mnist_example_utils
