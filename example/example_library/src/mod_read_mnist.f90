module read_mnist
  use constants_mnist, only: real32
  use athena, only: pad_data
  implicit none

  private

  public :: read_mnist_db

contains

!!!#############################################################################
!!! read mnist dataset
!!!#############################################################################
  subroutine read_mnist_db(file,images,labels,kernel_size,image_size,padding_method)
    use misc_mnist, only: icount
    implicit none
    integer :: i, j, k, Reason, unit
    integer :: num_samples, num_pixels, t_kernel_size = 1
    character(2048) :: buffer
    character(:), allocatable :: t_padding_method
    real(real32), allocatable, dimension(:,:,:,:) :: images_padded

    integer, intent(out) :: image_size
    integer, optional, intent(in) :: kernel_size
    character(*), optional, intent(in) :: padding_method
    character(1024), intent(in) :: file
    real(real32), allocatable, dimension(:,:,:,:), intent(out) :: images
    integer, allocatable, dimension(:), intent(out) :: labels


!!!-----------------------------------------------------------------------------
!!! initialise optional arguments
!!!-----------------------------------------------------------------------------
    if(present(padding_method))then
       t_padding_method = padding_method
    else
       t_padding_method = "valid"
    end if


!!!-----------------------------------------------------------------------------
!!! open file
!!!-----------------------------------------------------------------------------
    open(newunit=unit,file=file)


!!!-----------------------------------------------------------------------------
!!! count number of samples
!!!-----------------------------------------------------------------------------
    i = 0
    num_pixels = 0
    line_count: do
       i = i + 1
       read(unit,'(A)',iostat=Reason) buffer
       if(Reason.ne.0)then
          num_samples = i - 1
          exit line_count
       elseif(i.gt.90000)then
          write(0,*) "Too many lines to read in file provided (over 90000)"
          write(0,*) "Exiting..."
          stop 0
       elseif(i.eq.1)then
          num_pixels = icount(buffer,",") - 1
       end if
    end do line_count
    if(num_pixels.eq.0)then
       stop "Could not determine number of pixels"
    end if


!!!-----------------------------------------------------------------------------
!!! calculate size of image
!!!-----------------------------------------------------------------------------
    image_size = nint(sqrt(real(num_pixels,real32)))


!!!-----------------------------------------------------------------------------
!!! rewind file and allocate labels
!!!-----------------------------------------------------------------------------
    rewind(unit)
    if(allocated(labels)) deallocate(labels)
    allocate(labels(num_samples), source=0)


!!!-----------------------------------------------------------------------------
!!! allocate data set
!!! ... if appropriate, add padding
!!!-----------------------------------------------------------------------------
    !! dim=1: image width in pixels
    !! dim=2: image height in pixels
    !! dim=3: image number of channels (1 due to black-white images)
    !! dim=4: number of images
    if(.not.present(kernel_size))then
       stop "ERROR: kernel_size not provided to read_mnist for padding &
            &method "//t_padding_method
    else
       t_kernel_size = kernel_size
    end if
    
    if(allocated(images)) deallocate(images)
    allocate(images(image_size, image_size, 1, num_samples))


!!!-----------------------------------------------------------------------------
!!! read in dataset
!!!-----------------------------------------------------------------------------
    do i=1,num_samples
       read(unit,*) labels(i), ((images(j,k,1,i),k=1,image_size),j=1,image_size)
    end do

    close(unit)


!!!-----------------------------------------------------------------------------
!!! populate padding
!!!-----------------------------------------------------------------------------
    call pad_data(images, images_padded, &
         t_kernel_size, t_padding_method, 4, 3)
    images = images_padded
    

!!!-----------------------------------------------------------------------------
!!! increase label values to match fortran indices
!!!-----------------------------------------------------------------------------
    images = images/255._real32
    labels = labels + 1
    write(6,*) "Data read"

    return
  end subroutine read_mnist_db
!!!#############################################################################


end module read_mnist
